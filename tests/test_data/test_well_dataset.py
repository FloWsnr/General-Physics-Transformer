import pytest
from pathlib import Path

from gphyt.data.well_dataset import WellDataset


def test_well_dataset(dummy_datapath: Path):
    """Test the WellDataset class with a dummy dataset"""
    dataset = WellDataset(str(dummy_datapath.parent))
    assert len(dataset) > 0
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    # Check tensors are returned with correct dimensions
    assert x.ndim == 4  # (T, H, W, C)
    assert y.ndim == 4  # (T, H, W, C)


def test_well_dataset_with_include_filters(dummy_datapath: Path):
    """Test WellDataset with include_filters"""
    dataset = WellDataset(
        str(dummy_datapath.parent),
        include_filters=["field1"],
    )
    assert len(dataset) > 0
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    assert x.ndim == 4
    assert y.ndim == 4


def test_well_dataset_all_fields(dummy_datapath: Path):
    """Test the WellDataset class with all fields (no filters)"""
    dataset = WellDataset(str(dummy_datapath.parent))
    assert len(dataset) > 0
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    assert x.ndim == 4
    assert y.ndim == 4
