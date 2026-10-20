"""Shared loss helpers for SD3 RAM training."""


def adapter_host(model):
    """Return the (DDP-unwrapped) module exposing PEFT adapter controls."""
    if hasattr(model, "disable_adapter"):
        return model
    if hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
        return model.module
    raise AttributeError("Transformer does not expose PEFT adapter controls")


def model_forward(model, sample, timestep, prompt_embeds, pooled_embeds, accelerator):
    with accelerator.autocast():
        out = model(
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    return out.float()
