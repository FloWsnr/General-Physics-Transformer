import pytest
from pathlib import Path

from the_well.utils.dummy_data import write_dummy_data
from the_well.data.datasets import WellDataset


@pytest.fixture(scope="session")
def dummy_datapath(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create dummy data for testing."""
    data_dir = tmp_path_factory.mktemp("train")
    file = data_dir / "dummy_dataset.hdf5"
    write_dummy_data(file)
    return file


@pytest.fixture(scope="session")
def dummy_dataset(dummy_datapath: Path) -> WellDataset:
    """Create a dummy dataset for testing."""
    return WellDataset(path=dummy_datapath.parent)
