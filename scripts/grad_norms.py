"""Gradient norms for additive loss terms under accumulation, DDP, and FP16."""

from itertools import combinations

import torch


def validate_monitor_interval(interval):
    """Validate the epoch interval, where zero disables monitoring."""
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError(
            "grad_norm_monitor_every_n_epochs must be an integer >= 0, "
            f"got {interval!r}"
        )
    return interval


def should_monitor_epoch(interval, epoch):
    """Return whether gradient components should be measured this epoch."""
    return interval > 0 and epoch % interval == 0


def ram_residual_terms(output, reward_update, reverse_kl_anchor, integral_kl_anchor=None):
    """Decompose ``output.detach() - target_velocity`` into RAM terms.

    The RAM target is ``reward_update + integral_kl_anchor +
    reverse_kl_anchor``. Passing ``integral_kl_anchor=None`` omits the optional
    DMD term.
    """
    terms = {
        "ram_reference_anchor": output.detach() - reverse_kl_anchor.detach(),
        "ram_rwr": -reward_update.detach(),
    }
    if integral_kl_anchor is not None:
        terms["ram_dmd"] = -integral_kl_anchor.detach()

    for name, term in terms.items():
        if term.shape != output.shape:
            raise ValueError(f"{name} has shape {term.shape}, expected {output.shape}")
    return terms


def mse_vjp_surrogates(output, residual_terms, expected_residual):
    """Return scalars whose gradients are ``2 J^T residual / output.numel()``.

    ``residual_terms`` must add up to ``output.detach() - target``. The scalar
    values are not losses to log; they only produce the desired VJPs.
    """
    if not residual_terms:
        raise ValueError("At least one residual term is required")

    reconstructed = torch.zeros_like(expected_residual)
    surrogates = {}
    for name, residual in residual_terms.items():
        if residual.shape != output.shape:
            raise ValueError(f"{name} has shape {residual.shape}, expected {output.shape}")
        residual = residual.detach()
        reconstructed.add_(residual)
        surrogates[name] = (2.0 * output * residual).mean()

    error = torch.linalg.vector_norm((reconstructed - expected_residual.detach()).float())
    scale = sum(torch.linalg.vector_norm(term.detach().float()) for term in residual_terms.values())
    relative_error = error / scale.clamp_min(torch.finfo(torch.float32).eps)
    # A wrong decomposition is off by O(1); mixed-precision rounding alone reached
    # 1.02e-5 in a run without the reverse-KL anchor (whose norm otherwise
    # dominates the scale), so the tolerance leaves an order of magnitude of room.
    if relative_error.item() > 1e-4:
        raise ValueError(f"Residual terms do not reconstruct the MSE residual ({relative_error.item():.3e})")
    return surrogates


class GradNormMonitor:
    """Capture scaled component VJPs and report optimizer-step gradient metrics.

    Monitored forward/backward calls must run inside ``no_sync(model)`` because
    each component performs an additional backward through the same DDP graph.
    ``finish`` manually restores the DDP-averaged real gradient before the
    optimizer step.
    """

    def __init__(self, parameters, accelerator):
        # flattened represenation of all monitored parameter gradients
        self.parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not self.parameters:
            raise ValueError("No trainable parameters to monitor")
        if len({id(parameter) for parameter in self.parameters}) != len(self.parameters):
            raise ValueError("Duplicate parameters are not supported")
        if len({parameter.device for parameter in self.parameters}) != 1:
            raise ValueError("All monitored parameters must be on one device")

        self.accelerator = accelerator
        self.device = self.parameters[0].device
        self.num_parameters = sum(parameter.numel() for parameter in self.parameters)
        # offsets for indexing each parameter inside self.parameters
        self.offsets = []
        offset = 0
        for parameter in self.parameters:
            self.offsets.append((offset, offset + parameter.numel()))
            offset += parameter.numel()

        # Reuse zeroed FP32 buffers across optimizer steps to avoid allocation churn;
        # terms_seen distinguishes components captured in the current step.
        self.buffers = {}
        self.terms_seen = set()
        self.loss_scale = None

    def no_sync(self, model):
        """Disable automatic DDP reduction for a monitored forward/backward."""
        return self.accelerator.no_sync(model)

    def _flat_buffer(self):
        return torch.zeros(self.num_parameters, device=self.device, dtype=torch.float32)

    def capture(self, terms):
        """Accumulate directly measured component gradients for one microbatch."""
        accumulation_steps = self.accelerator.gradient_accumulation_steps
        for name, term in terms.items():
            if name not in self.buffers:
                self.buffers[name] = self._flat_buffer()
            self.terms_seen.add(name)

            component = term / accumulation_steps
            if self.accelerator.scaler is not None:
                component = self.accelerator.scaler.scale(component)
                current_scale = float(self.accelerator.scaler.get_scale())
            else:
                current_scale = 1.0
            if self.loss_scale is None:
                self.loss_scale = current_scale
            elif current_scale != self.loss_scale:
                raise RuntimeError("GradScaler scale changed within one optimizer step")

            handles = []
            for parameter, (start, end) in zip(self.parameters, self.offsets, strict=True):
                destination = self.buffers[name][start:end]

                def save_gradient(gradient, destination=destination):
                    destination.add_(gradient.detach().reshape(-1).float())
                    return torch.zeros_like(gradient)

                handles.append(parameter.register_hook(save_gradient))

            try:
                component.backward(retain_graph=True)
            finally:
                for handle in handles:
                    handle.remove()

    def _flatten_real_gradient(self):
        flat = self._flat_buffer()
        for parameter, (start, end) in zip(self.parameters, self.offsets, strict=True):
            if parameter.grad is not None:
                flat[start:end].copy_(parameter.grad.detach().reshape(-1).float())
        return flat

    def _copy_real_gradient(self, flat):
        for parameter, (start, end) in zip(self.parameters, self.offsets, strict=True):
            if parameter.grad is not None:
                parameter.grad.copy_(flat[start:end].view_as(parameter).to(parameter.grad.dtype))

    def finish(self, optimizer):
        """Synchronize, unscale, measure, and reset one optimizer-step capture."""
        if not self.accelerator.sync_gradients:
            raise RuntimeError("finish must be called at an optimizer-step boundary")
        if not self.terms_seen or self.loss_scale is None:
            raise RuntimeError("No component gradients were captured")

        # Match ordinary FP16 DDP: average the scaled real gradient first, then
        # let GradScaler inspect and unscale that globally averaged gradient.
        real_scaled = self.accelerator.reduce(self._flatten_real_gradient(), reduction="mean")
        self._copy_real_gradient(real_scaled)
        self.accelerator.unscale_gradients(optimizer)
        combined = self._flatten_real_gradient()

        vectors = {}
        for name in sorted(self.terms_seen):
            vector = self.accelerator.reduce(self.buffers[name], reduction="mean")
            vectors[name] = vector.div_(self.loss_scale)

        component_sum = torch.zeros_like(combined)
        for vector in vectors.values():
            component_sum.add_(vector)

        norms = {name: torch.linalg.vector_norm(vector).item() for name, vector in vectors.items()}
        combined_norm = torch.linalg.vector_norm(combined).item()
        difference_norm = torch.linalg.vector_norm(component_sum - combined).item()
        reconstruction_scale = max(combined_norm, sum(norms.values()), torch.finfo(torch.float32).eps)

        metrics = {
            "grad_norm/combined": combined_norm,
            "grad_monitor/reconstruction_rel_error": difference_norm / reconstruction_scale,
            "grad_monitor/loss_scale": self.loss_scale,
            "grad_monitor/buffer_mib": 4 * self.num_parameters * len(vectors) / 1024**2,
        }
        metrics.update({f"grad_norm/{name}": norm for name, norm in norms.items()})
        for first, second in combinations(sorted(vectors), 2):
            denominator = norms[first] * norms[second]
            cosine = torch.dot(vectors[first], vectors[second]).item() / denominator if denominator else 0.0
            metrics[f"grad_cos/{first}__{second}"] = cosine

        for buffer in self.buffers.values():
            buffer.zero_()
        self.terms_seen.clear()
        self.loss_scale = None
        return metrics

    def release(self):
        """Release persistent component buffers."""
        self.buffers.clear()
        self.terms_seen.clear()
        self.loss_scale = None
