from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from gphyt.train.train_base import Trainer, TrainingState
from gphyt.models.loss_fns import RMSE


@pytest.fixture
def real_model() -> nn.Module:
    """Create a real PyTorch model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor([1.0]))
            self.layer = nn.Identity()

        def forward(self, x):
            return self.layer(x) + self.param

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel()
    model.to(device)
    return model


@pytest.fixture
def real_optimizer(real_model: nn.Module) -> torch.optim.Optimizer:
    """Create a real PyTorch optimizer for testing."""
    return torch.optim.SGD(real_model.parameters(), lr=0.001)


@pytest.fixture
def real_criterion() -> nn.Module:
    """Create a real PyTorch loss function for testing."""
    return nn.MSELoss()


@pytest.fixture
def real_lr_scheduler(
    real_optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a real PyTorch learning rate scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(real_optimizer, step_size=10, gamma=0.1)


@pytest.fixture
def real_dataloader() -> DataLoader:
    """Create a real PyTorch DataLoader for testing."""
    # Create dummy data in the format expected by trainer
    input_data = torch.randn(4, 10, 10)
    target_data = torch.randn(4, 10, 10)

    # Create dataset with proper format
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return (self.inputs[idx], self.targets[idx])

    dataset = TestDataset(input_data, target_data)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for testing."""
    return tmp_path / "test_output"


@pytest.fixture
def real_scaler() -> GradScaler:
    """Create a real PyTorch GradScaler for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GradScaler(device=device, enabled=True)


class TestTrainingState:
    """Test the TrainingState dataclass."""

    def test_training_state_initialization(self):
        """Test TrainingState initialization with defaults."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.samples_trained == 0
        assert state.batches_trained == 0
        assert state.current_lr == 0.0
        assert state.batch_size == 0
        assert not state.shutdown.item()

    def test_training_state_custom_values(self):
        """Test TrainingState initialization with custom values."""
        state = TrainingState(
            epoch=5,
            samples_trained=1000,
            batches_trained=50,
            current_lr=0.01,
            batch_size=32,
        )
        assert state.epoch == 5
        assert state.samples_trained == 1000
        assert state.batches_trained == 50
        assert state.current_lr == 0.01
        assert state.batch_size == 32


class TestTrainer:
    """Test the Trainer class."""

    def test_trainer_initialization(
        self,
        real_model,
        real_optimizer,
        real_criterion,
        real_lr_scheduler,
        real_dataloader,
        temp_output_dir,
        real_scaler,
    ):
        """Test Trainer initialization."""
        trainer = Trainer(
            model=real_model,
            optimizer=real_optimizer,
            criterion=real_criterion,
            lr_scheduler=real_lr_scheduler,
            train_dataloader=real_dataloader,
            val_dataloader=real_dataloader,
            scaler=real_scaler,
            total_updates=100,
            updates_per_epoch=10,
            checkpoint_every_updates=50,
            eval_fraction=1.0,
            epoch=1,
            batches_trained=0,
            samples_trained=0,
            output_dir=temp_output_dir,
            loss_fns={"RMSE": RMSE(dims=None)},
            amp=True,
            amp_precision=torch.bfloat16,
            max_grad_norm=None,
            time_limit=None,
            wandb_logger=None,
            global_rank=0,
            local_rank=0,
            world_size=1,
        )

        assert trainer.model == real_model
        assert trainer.optimizer == real_optimizer
        assert trainer.criterion == real_criterion
        assert trainer.lr_scheduler == real_lr_scheduler
        assert trainer.train_dataloader == real_dataloader
        assert trainer.val_dataloader == real_dataloader
        assert trainer.total_updates == 100
        assert trainer.updates_per_epoch == 10
        assert trainer.output_dir == temp_output_dir
        assert trainer.wandb_logger is None
        assert trainer.global_rank == 0
        assert trainer.local_rank == 0
        assert trainer.world_size == 1

        # Check output directory was created
        assert temp_output_dir.exists()

        # Check state initialization
        assert trainer.state.epoch == 1
        assert trainer.state.samples_trained == 0
        assert trainer.state.batches_trained == 0
        assert trainer.state.current_lr == 0.001
        assert trainer.state.batch_size == 2  # dataloader batch_size * world_size

        # Check loss functions
        assert isinstance(trainer.loss_fns["RMSE"], RMSE)

    def test_train_updates(
        self,
        real_model: nn.Module,
        real_optimizer: torch.optim.Optimizer,
        real_criterion: nn.Module,
        real_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        real_dataloader: DataLoader,
        temp_output_dir: Path,
        real_scaler: GradScaler,
    ):
        """Test train_updates method."""

        trainer = Trainer(
            model=real_model,
            optimizer=real_optimizer,
            criterion=real_criterion,
            lr_scheduler=real_lr_scheduler,
            train_dataloader=real_dataloader,
            val_dataloader=real_dataloader,
            scaler=real_scaler,
            total_updates=100,
            updates_per_epoch=10,
            checkpoint_every_updates=50,
            eval_fraction=1.0,
            epoch=1,
            batches_trained=0,
            samples_trained=0,
            output_dir=temp_output_dir,
            loss_fns={"RMSE": RMSE(dims=None)},
        )

        initial_batches = trainer.state.batches_trained
        initial_samples = trainer.state.samples_trained

        # Run training updates
        trainer.train_updates(n_updates=1, epoch=1)

        # Verify model is in training mode
        assert real_model.training

        # Verify state updates
        assert trainer.state.batches_trained == initial_batches + 1
        assert (
            trainer.state.samples_trained == initial_samples + trainer.state.batch_size
        )

    def test_validate(
        self,
        real_model,
        real_optimizer,
        real_criterion,
        real_lr_scheduler,
        real_dataloader,
        temp_output_dir,
        real_scaler,
    ):
        """Test validate method."""
        trainer = Trainer(
            model=real_model,
            optimizer=real_optimizer,
            criterion=real_criterion,
            lr_scheduler=real_lr_scheduler,
            train_dataloader=real_dataloader,
            val_dataloader=real_dataloader,
            scaler=real_scaler,
            total_updates=100,
            updates_per_epoch=10,
            checkpoint_every_updates=50,
            eval_fraction=1.0,
            epoch=1,
            batches_trained=0,
            samples_trained=0,
            output_dir=temp_output_dir,
            loss_fns={"RMSE": RMSE(dims=None)},
        )

        # Run validation
        trainer.validate(epoch=1)

    def test_run_method_basic(
        self,
        real_model,
        real_optimizer,
        real_criterion,
        real_lr_scheduler,
        real_dataloader,
        temp_output_dir,
        real_scaler,
    ):
        """Test run method basic functionality."""
        trainer = Trainer(
            model=real_model,
            optimizer=real_optimizer,
            criterion=real_criterion,
            lr_scheduler=real_lr_scheduler,
            train_dataloader=real_dataloader,
            val_dataloader=real_dataloader,
            scaler=real_scaler,
            total_updates=2,  # Small number for testing
            updates_per_epoch=1,
            checkpoint_every_updates=50,
            eval_fraction=1.0,
            epoch=1,
            batches_trained=0,
            samples_trained=0,
            output_dir=temp_output_dir,
            loss_fns={"RMSE": RMSE(dims=None)},
        )
        # Run training
        trainer.run()

    def test_time_limit_functionality(
        self,
        real_model,
        real_optimizer,
        real_criterion,
        real_lr_scheduler,
        real_dataloader,
        temp_output_dir,
        real_scaler,
    ):
        """Test time limit and shutdown functionality."""
        trainer = Trainer(
            model=real_model,
            optimizer=real_optimizer,
            criterion=real_criterion,
            lr_scheduler=real_lr_scheduler,
            train_dataloader=real_dataloader,
            val_dataloader=real_dataloader,
            scaler=real_scaler,
            total_updates=100,
            updates_per_epoch=10,
            checkpoint_every_updates=50,
            eval_fraction=1.0,
            epoch=1,
            batches_trained=0,
            samples_trained=0,
            output_dir=temp_output_dir,
            loss_fns={"RMSE": RMSE(dims=None)},
            time_limit=0,  # no time limit
        )
        # Run training
        trainer.run()

        # Verify shutdown was triggered
        assert trainer.state.shutdown.item()
