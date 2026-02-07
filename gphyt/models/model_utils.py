import torch


def get_model(model_config: dict) -> torch.nn.Module:
    """Factory function to create a model based on configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration dictionary. Must contain 'architecture' key.

    Returns
    -------
    torch.nn.Module
        The instantiated model.
    """
    architecture = model_config.get("architecture", "gphyt")

    if architecture == "gphyt":
        from gphyt.models.transformer.model import get_model as get_gpt_model
        return get_gpt_model(model_config)
    elif architecture == "unet":
        from gphyt.models.unet import get_model as get_unet_model
        return get_unet_model(model_config)
    elif architecture == "fno":
        from gphyt.models.fno import get_model as get_fno_model
        return get_fno_model(model_config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
