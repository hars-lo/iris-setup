from iris_ml.data.factory import register_dataset

# Import dataset adapters so they register themselves
from .acdc import ACDCDataset
from .amos import AMOSDataset
from .msd_pancreas import MSDPancreasDataset
from .segthor import SegTHORDataset

__all__ = [
    "ACDCDataset",
    "AMOSDataset",
    "MSDPancreasDataset",
    "SEGTHORDataset"
]
