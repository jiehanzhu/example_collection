import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LiftChart:
    def __init__(self, y: np.array, yhat: np.array):
        """ Create the dataset for lift chart creation"""
        exposure = [1.0] * len(y)
        self.data = pd.DataFrame(
            {
                "actual_y": y,
                "predicted_y": yhat * (sum(y * exposure) / sum(yhat * exposure)),  # rebase
                "mean_y": [np.average(y)] * len(y),
                "exposure": exposure,
            }
        )
        self.data["actual_y"] = self.data["actual_y"] * self.data["exposure"]
        self.data["predicted_y"] = self.data["predicted_y"] * self.data["exposure"]
        self.data.sort_values(by="predicted_y", inplace=True)
        self.data["cumul_exposure"] = np.cumsum(self.data["exposure"]) / np.sum(
            self.data["exposure"]
        )
        self.data["residual"] = (self.data["predicted_y"] - self.data["actual_y"])
        self.data["ss_total"] = (self.data["actual_y"] - self.data["mean_y"]).pow(2)
        self.data["ss_residual"] = (self.data["residual"]).pow(2)

    def append_bucket(self, n_quantiles: int):
        # Create the quantiles using exposure_frac
        bin_width = 1 / float(n_quantiles)
        self.data["bin"] = self.data["cumul_exposure"].apply(lambda x: x // bin_width)

    def quantile_data_r_square(self, n_quantiles: int = 10):
        """
        Use the whole dataset mean and whole dataset as the denominator, but only include quantile data in numerator.
        """
        self.append_bucket(n_quantiles)
        data_agg = self.data.groupby("bin")[["ss_total", "ss_residual", "exposure"]].sum().reset_index()
        data_agg["exposure"] /= sum(data_agg["exposure"])

        data_agg["r_square"] = (data_agg["ss_total"] - data_agg["ss_residual"]) / sum(data_agg["ss_total"])
        data_agg["cumul_r_square"] = 1 - np.cumsum(data_agg["ss_residual"]) / sum(data_agg["ss_total"])
        return data_agg[["bin", "exposure", 'r_square', "cumul_r_square"]].to_dict('list')

    def quantile_data_lift(self, n_quantiles: int = 10):
        """
        Divides data into n_quantiles, calculating the mean for each
        quantile. The data are split using the cumulated exposure, ensuring that
        each quantile has a balanced amount of exposure. This is described in
        Generalized Linear Models for Insurance Rating by Goldburd, Khare, Tevet.
        """
        self.append_bucket(n_quantiles)
        data_agg = self.data.groupby("bin")[["actual_y", "predicted_y", "exposure"]].sum().reset_index()
        for col in ["actual_y", "predicted_y"]:
            data_agg[col] = data_agg[col] / data_agg["exposure"]
        data_agg["exposure"] /= sum(data_agg["exposure"])
        return data_agg[["bin", "exposure", "actual_y", "predicted_y"]].to_dict('list')

    def quantile_plot(self, n_quantiles: int = 10, chart: str = "lift") -> plt.Figure:
        """
        Plot or save the lift chart with selected number of buckets.
        """
        if chart == "lift":
            quantile_dict = self.quantile_data_lift(n_quantiles)
            columns_label = {
                "title":"Lift Chart",
                "line_1": "actual_y",
                "line_1_name": "True Target",
                "line_2": "predicted_y",
                "line_2_name": "Predicted Target",
                "y_label": "Average Target in Quantile Range"
            }
        elif chart == "r_square":
            quantile_dict = self.quantile_data_r_square(n_quantiles)
            columns_label = {
                "title": "R Square Chart",
                "line_1": "r_square",
                "line_1_name": "Quantile R Square",
                "line_2": "cumul_r_square",
                "line_2_name": "Cumulative R Square",
                "y_label": "R Square in Quantile Range"
            }
        else:
            raise ValueError("chart must be either 'lift' or 'r_square'")

        x = [(i + 0.5) / n_quantiles for i in range(n_quantiles)]

        # create plot
        fig, ax = plt.subplots(figsize=(12, 9))

        ax.plot(x, quantile_dict[columns_label["line_1"]], ls="-", lw=3, label=columns_label["line_1_name"])
        clr = ax.lines[-1].get_color()
        ax.plot(
            x,
            quantile_dict[columns_label["line_2"]],
            ls="--",
            marker=".",
            markersize=10,
            color=clr,
            label=columns_label["line_2_name"],
        )
        ax_dup = ax.twinx()
        ax_dup.bar(
            x,
            quantile_dict["exposure"],
            width=(0.5 / n_quantiles),
            align="edge",
            alpha=0.5,
            label="Exposure",
        )
        ax.set_xlabel("Quantile", size=14)
        ax.set_ylabel(columns_label["y_label"], size=14)
        ax_dup.set_ylabel("Exposure", size=14)
        ax_dup.set_ylim([0, 1 / (n_quantiles // 3)])
        ax.set_title(columns_label["title"], size=18)
        ax.legend()
        ax_dup.legend()
        return ax.figure
