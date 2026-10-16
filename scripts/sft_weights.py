"""Built-in SFT loss weight schedules for SD3 training."""


def linear(epoch, base_weight, decay_until_epoch):
    """Linearly decay ``base_weight`` to zero by ``decay_until_epoch``."""
    if not isinstance(decay_until_epoch, int):
        raise ValueError("linear SFT weight decay epoch must be an integer")
    if decay_until_epoch <= 0:
        raise ValueError("linear SFT weight decay epoch must be positive")
    return float(base_weight) * max(0.0, 1.0 - epoch / decay_until_epoch)


def linear_after(epoch, base_weight, start_decay_epoch, end_decay_epoch):
    """Hold ``base_weight`` through ``start_decay_epoch``, then decay to zero."""
    if not isinstance(start_decay_epoch, int) or not isinstance(end_decay_epoch, int):
        raise ValueError("linear_after SFT weight epochs must be integers")
    if start_decay_epoch < 0 or start_decay_epoch >= end_decay_epoch:
        raise ValueError("linear_after requires 0 <= start_decay_epoch < end_decay_epoch")
    if epoch <= start_decay_epoch:
        return float(base_weight)
    if epoch >= end_decay_epoch:
        return 0.0
    decay_progress = (epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
    return float(base_weight) * (1.0 - decay_progress)


def resolve(spec, epoch, base_weight, config):
    """Resolve an SFT loss weight schedule spec.

    Supported specs:
        ["linear", decay_until_epoch]
        ["linear_after", start_decay_epoch, end_decay_epoch]
    """
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError("sft_loss_weight_fn must be [linear, N] or [linear_after, START, END]")

    name = spec[0]
    if name == "linear":
        if len(spec) != 2:
            raise ValueError("linear SFT weight schedule must be [linear, decay_until_epoch]")
        return linear(epoch, base_weight, spec[1])
    if name == "linear_after":
        if len(spec) != 3:
            raise ValueError("linear_after SFT weight schedule must be [linear_after, START, END]")
        return linear_after(epoch, base_weight, spec[1], spec[2])
    raise NotImplementedError(f"Unsupported sft_loss_weight_fn: {name}")
