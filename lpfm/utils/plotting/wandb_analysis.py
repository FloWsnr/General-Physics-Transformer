import wandb
from wandb.apis.public import Run
import dotenv
import pandas as pd

from lpfm.utils.plotting.base_plotter import BasePlotter

dotenv.load_dotenv()


class WandbPlotter:
    def __init__(self, run_ids: list[str]):
        self.api = wandb.Api()
        entity, project = "wsnr-florian", "Large-Physics-Foundation-Model"
        self.runs: list[Run] = self.api.runs(
            f"{entity}/{project}", {"$or": [{"name": run_id} for run_id in run_ids]}
        )
        print(f"Found {len(self.runs)} runs")

    def plot_training_loss(self):
        # get losses from all runs\
        keys = ["training-losses/NMSE", "training/total_samples_trained"]
        data = {}
        for run in self.runs:
            data[run.name] = run.history(keys=keys, pandas=True)

        # create a multiindex df from the data
        df = pd.concat(data, axis=1)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        plotter = BasePlotter(
            color="white",
        )

        # get the max number of samples trained
        max_samples = (
            df.xs("training/total_samples_trained", axis=1, level=1).max().max()
        )

        plotter.setup_figure(
            x_ticks=[0, max_samples // 2, max_samples],
            y_ticks=[0.001, 0.1, 10],
            x_label="Samples Trained",
            y_label="NMSE",
            y_log=True,
        )

        # plot the data
        for run_name in data.keys():
            samples = df[run_name]["training/total_samples_trained"]
            loss = df[run_name]["training-losses/NMSE"]
            color = next(plotter.color_cycler)
            plotter.plot_data(
                samples, loss, label=run_name, symbolstyle="", color=color
            )

        plotter.legend(title="Models", loc="upper right")
        plotter.show_figure()


if __name__ == "__main__":
    plotter = WandbPlotter(
        ["ti-cyl-sym-flow-0001", "ti-cyl-sym-flow-0001b", "ti-cyl-sym-flow-0001c"]
    )
    plotter.plot_training_loss()
