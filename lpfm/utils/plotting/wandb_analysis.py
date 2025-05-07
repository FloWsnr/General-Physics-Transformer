import wandb
from wandb.apis.public import Run
import dotenv
import pandas as pd

from lpfm.utils.plotting.base_plotter import BasePlotter

dotenv.load_dotenv()


class WandbLoader:
    def __init__(self, run_ids: list[str]):
        self.api = wandb.Api()
        entity, project = "wsnr-florian", "Large-Physics-Foundation-Model"
        self.runs: list[Run] = self.api.runs(
            f"{entity}/{project}", {"$or": [{"name": run_id} for run_id in run_ids]}
        )


class LossPlotter(WandbLoader):
    def __init__(self, run_ids: list[str]):
        super().__init__(run_ids)
        print(f"Found {len(self.runs)} runs")

        keys = [
            "training-losses/NMSE",
            "training/total_samples_trained",
            "training/total_batches_trained",
        ]
        train_data = {}
        for run in self.runs:
            train_data[run.name] = run.history(keys=keys, pandas=True)
        self.train_data = pd.concat(train_data, axis=1)
        self.train_data.columns = pd.MultiIndex.from_tuples(self.train_data.columns)

        keys = [
            "validation-summary/samples_trained",
            "validation-sumamry/total-NMSE",
        ]
        val_data = {}
        for run in self.runs:
            val_data[run.name] = run.history(keys=keys, pandas=True)
        self.val_data = pd.concat(val_data, axis=1)
        self.val_data.columns = pd.MultiIndex.from_tuples(self.val_data.columns)

        self.y_ticks = [0.0001, 0.01, 1]

    def loss_over_samples(self):
        plotter = BasePlotter(
            color="white",
        )

        # get the max number of samples trained
        max_samples = (
            self.train_data.xs("training/total_samples_trained", axis=1, level=1)
            .max()
            .max()
        )

        plotter.setup_figure(
            x_ticks=[0, max_samples // 2, max_samples],
            y_ticks=self.y_ticks,
            x_label="Samples Trained",
            y_label="NMSE",
            y_log=True,
        )

        # plot the data
        for run_name in self.train_data.columns.get_level_values(0):
            samples = self.train_data[run_name]["training/total_samples_trained"]
            loss = self.train_data[run_name]["training-losses/NMSE"]
            color = next(plotter.color_cycler)
            plotter.plot_data(
                samples, loss, label=run_name, symbolstyle="", color=color
            )

        plotter.legend(title="Models", loc="upper right")
        plotter.show_figure()

    def loss_over_batches(self):
        plotter = BasePlotter(
            color="white",
        )

        # get the max number of batches trained
        max_batches = (
            self.train_data.xs("training/total_batches_trained", axis=1, level=1)
            .max()
            .max()
        )

        plotter.setup_figure(
            x_ticks=[0, max_batches // 2, max_batches],
            y_ticks=self.y_ticks,
            x_label="Batches Trained",
            y_label="NMSE",
            y_log=True,
        )

        # plot the data
        for run_name in self.train_data.columns.get_level_values(0):
            batches = self.train_data[run_name]["training/total_batches_trained"]
            loss = self.train_data[run_name]["training-losses/NMSE"]
            color = next(plotter.color_cycler)
            plotter.plot_data(
                batches, loss, label=run_name, symbolstyle="", color=color
            )

        plotter.legend(title="Models", loc="upper right")
        plotter.show_figure()


if __name__ == "__main__":
    plotter = LossPlotter(
        ["ti-cyl-sym-flow-0001", "ti-cyl-sym-flow-0001b", "ti-cyl-sym-flow-0001c"]
    )
    plotter.loss_over_samples()
    plotter.loss_over_batches()
