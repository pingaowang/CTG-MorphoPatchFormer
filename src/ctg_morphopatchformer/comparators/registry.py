"""Architecture profiles selected from the available experimental configurations."""
from __future__ import annotations

from copy import deepcopy

MODEL_NAMES = ("CTG-MorphoPatchFormer", "CTGNet", "EMAUNet", "ETCNN", "LCResU",
               "MSEU", "MTU-Net3+", "WaveFHR-VNet", "WMFB")
_PARAMETERS = {
    "CTG-MorphoPatchFormer": {},
    "CTGNet": {"base_channels": 32, "kernel_size": 11, "dropout": 0.1,
               "multiscale": False, "eca": False},
    "EMAUNet": {"base_channels": 16, "kernel_sizes": (21, 31, 61, 81),
                "dropout": 0.1, "use_eca": True, "use_nla": True, "fusion": "or_logits"},
    "ETCNN": {"base_channels": 8, "kernel_sizes": (5, 9, 15, 21),
              "dropout": 0.1, "use_cct": True, "cct_patch_size": 48,
              "cct_heads": 4, "cct_layers": 5, "fusion": "or_logits"},
    "LCResU": {"base_channels": 8, "kernel_size": 5, "dropout": 0.0},
    "MSEU": {"base_channels": 32, "kernel_sizes": (7, 15, 21),
             "base_kernel_size": 17, "mse_fusion": "concat_only", "dropout": 0.1},
    # Reference profile only: the historical faithful_v3 run binding is unresolved.
    "MTU-Net3+": {"base_channels": 32, "dropout": 0.1, "deep_supervision": True},
    "WaveFHR-VNet": {"base_channels": 64, "dropout": 0.1, "ltcm_heads": 8},
}


def inspection_parameters(name: str) -> dict:
    """Return structural parameters; see README.md for the MTU provenance limit."""
    if name == "WMFB":
        return {"fs_hz": 4.0}
    if name not in _PARAMETERS:
        raise ValueError(f"Unknown method {name!r}; choose from {MODEL_NAMES}")
    return deepcopy(_PARAMETERS[name])


def build_model(name: str, **overrides):
    """Construct an untrained architecture in evaluation mode.

    Explicit overrides are for inspection, not automatically paper-result presets.
    WMFB has no network: use the separately licensed third_party/wmfb module.
    """
    if name == "WMFB":
        raise ValueError("WMFB is deterministic; see third_party/wmfb/README.md.")
    params = inspection_parameters(name)
    params.update(overrides)
    if name == "CTG-MorphoPatchFormer":
        from ..model import MorphoPatchFormer
        cls = MorphoPatchFormer
    elif name == "CTGNet":
        from .unet1d import CtgNet1D
        cls = CtgNet1D
    elif name == "EMAUNet":
        from .emaunet1d import EmaUNet1D
        cls = EmaUNet1D
    elif name == "ETCNN":
        from .etcnn1d import ETCNN1D
        cls = ETCNN1D
    elif name == "LCResU":
        from .lcresunet1d import LCResUNet1D
        cls = LCResUNet1D
    elif name == "MSEU":
        from .mseunet1d import MSEUNet1D
        cls = MSEUNet1D
    elif name == "MTU-Net3+":
        from .mtunet3plus1d import MTUNet3Plus1D
        cls = MTUNet3Plus1D
    elif name == "WaveFHR-VNet":
        from .voting import WaveFHRHardVote1D
        cls = WaveFHRHardVote1D
    else:
        raise ValueError(f"Unknown method {name!r}")
    return cls(**params).eval()
