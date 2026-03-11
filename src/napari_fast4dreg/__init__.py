__version__ = "0.2.7"
__tag__ = "v0.27"

from ._fast4Dreg_functions import get_gpu_info, set_gpu_acceleration
from .api import fast4dreg, register, register_image, register_image_from_file

try:
    from ._widget import Fast4DReg_widget
except Exception:  # pragma: no cover - optional napari dependency
    def Fast4DReg_widget(*args, **kwargs):
        raise ImportError(
            "Fast4DReg_widget requires napari and Qt dependencies. "
            "Install with 'pip install napari-fast4dreg[napari]'."
        )

__all__ = (
    "Fast4DReg_widget",
    "register_image",
    "register_image_from_file",
    "fast4dreg",
    "register",
    "set_gpu_acceleration",
    "get_gpu_info",
)
