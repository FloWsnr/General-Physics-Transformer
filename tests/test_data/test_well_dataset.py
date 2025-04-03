import pytest
from pathlib import Path

from metaparc.data.well_dataset import WellDataset


def test_well_dataset(dummy_datapath: Path):
    """Test the WellDataset class with a dummy dataset and a custom field selection"""
    field_names = {
        "t0_fields": ["variable_field1"],
        "t1_fields": ["field1"],
    }

    dataset = WellDataset(str(dummy_datapath.parent), include_field_names=field_names)
    assert len(dataset) == 18
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    assert x.shape == (1, 32, 32, 3)
    assert y.shape == (1, 32, 32, 3)


def test_well_dataset_only_t1(dummy_datapath: Path):
    """Test the WellDataset class with a dummy dataset and a custom field selection"""
    field_names = {
        "t1_fields": ["field1"],
    }

    dataset = WellDataset(str(dummy_datapath.parent), include_field_names=field_names)
    assert len(dataset) == 18
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    assert x.shape == (1, 32, 32, 2)
    assert y.shape == (1, 32, 32, 2)


def test_well_dataset_all_fields(dummy_datapath: Path):
    """Test the WellDataset class with a dummy dataset and a custom field selection"""
    field_names = {}

    dataset = WellDataset(str(dummy_datapath.parent), include_field_names=field_names)
    assert len(dataset) == 18
    data = dataset[0]
    x = data["input_fields"]
    y = data["output_fields"]
    assert x.shape == (1, 32, 32, 6)
    assert y.shape == (1, 32, 32, 6)
