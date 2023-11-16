import tempfile
from typing import Any

import torch

from ember.utils import LoggingMixin


class EarlyStopping(LoggingMixin):
    """Implements early stopping in a Pytorch fashion, i.e. an init call
    where the model (you want to save) is an argument and a step function
    to be called after each evaluation.

    Attributes:
        model: nn.Module to be saved.
        tmp_fn: TemporaryFile, where to save model (can be None).
        patience: early stopping patience.
        cnt: number of early stopping steps that metric has not improved.
        delta: difference before new metric is considered better that the
            previous best one.
        higher_better: whether a higher metric is better.
        best: best metric value so far.
        best_<metric name>: other corresponding measurements can be passed
            as extra kwargs, they are stored when the main metric is stored
            by prepending 'best_' to the name.
    """

    @staticmethod
    def argparse_args():
        return dict(
            early_stopping_patience=dict(
                type=int, help="early stopping patience"
            ),
            early_stopping_metric=dict(
                type=str,
                help="metric to use for early stopping",
                metadata=dict(parent="early_stopping_patience"),
            ),
            early_stopping_delta=dict(
                type=float,
                help="difference before new metric is considered "
                "better that the previous best one",
                metadata=dict(parent="early_stopping_patience"),
            ),
            early_stopping_lower_better=dict(
                action="store_true",
                help="whether a lower metric is better",
                metadata=dict(parent="early_stopping_patience"),
            ),
        )

    def __init__(
        self,
        model: torch.nn.Module,
        patience: int | None,
        save_model: bool = False,
        delta: float = 0,
        lower_better: bool = False,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            model: nn.Module to be saved.
            save_model: whether to save model.
            patience: early stopping patience, if `None then no early stopping.
            delta: difference before new metric is considered better that
                the previous best one.
            lower_better: whether a lower metric is better.
        """

        super().__init__(*args, **kwargs)

        self.model = model
        self.tmp_fn = (
            tempfile.NamedTemporaryFile(mode="r+", suffix=".pt")
            if save_model
            else None
        )
        self.saved = False
        self.patience = patience
        self.cnt = 0
        self.delta = delta
        self.higher_better = not lower_better

        self.best = None

    def new_best(self, metric: float) -> bool:
        """Compares the `metric` appropriately to the current best.

        Args:
            metric: metric to compare best to.

        Returns:
            True if metric is indeed better, False otherwise.
        """
        if self.best is None:
            return True
        return (
            metric > self.best + self.delta
            if self.higher_better
            else metric < self.best - self.delta
        )

    def best_str(self) -> str:
        """Formats `best` appropriately."""
        if self.best is None:
            return "None"
        return f"{self.best:.6f}"

    def step(self, metric: float | None, **kwargs) -> bool:
        """Compares new metric (if it is provided) with previous best,
        saves model if so (and if `model_path` was not `None`) and
        updates count of unsuccessful steps.

        Args:
            metric: metric value based on which early stopping is used.
            kwargs: all desired metrics (including the metric passed).

        Returns:
            Whether the number of unsuccesful steps has exceeded the
            patience if patience has been set, else the signal to
            continue training (aka `False`).
        """
        if self.patience is None or metric is None:
            self._save()
            return False  # no early stopping, so user gets signal to continue

        if self.new_best(metric):
            self.log(
                f"Metric improved: {self.best_str()} -> {metric:.6f}", "info"
            )
            self._store_best(metric, **kwargs)
            self.cnt = 0
            self._save()
        else:
            self.cnt += 1
            self.log(
                f"Patience counter increased to {self.cnt}/{self.patience}",
                "info",
            )

        return self.cnt >= self.patience

    def _save(self):
        """Saves model and logs location."""
        if self.tmp_fn is not None:
            self.saved = True
            torch.save(self.model.state_dict(), self.tmp_fn.name)
            self.tmp_fn.seek(0)
            self.log("Saved model to " + self.tmp_fn.name, "info")

    def best_model(self) -> torch.nn.Module:
        """Loads last checkpoint (if any) and returns model."""
        if self.tmp_fn is not None and self.saved:
            state = torch.load(self.tmp_fn.name)
            self.model.load_state_dict(state)
        return self.model

    def _store_best(self, metric: float, **kwargs):
        """Saves best metric and potentially other corresponsing
        measurements in kwargs."""
        self.best = metric
        for key in kwargs:
            self.__setattr__("best_" + key, kwargs[key])

    def get_metrics(
        self,
    ) -> dict[str, Any] | None:
        """Returns accumulated best metrics.

        Returns:
            If the class was idle, nothing. Otherwise, if metrics were
            passed with kwargs in `step`, then these with the string
            `best_` prepended in their keys, else a generic dict
            with 'metric' as key and the best metric.
        """

        if self.best is None:
            return

        metrics = {
            k: v for k, v in self.__dict__.items() if k.startswith("best_")
        }

        if not metrics:
            metrics = {"metric": self.best}

        return metrics


class SchedulerList:
    """Helper to hold multiple schedulers. Schedulers should have
    a step method, with no required arguments.

    Attributes:
        schedulers: list of schedulers.
    """

    def __init__(
        self,
        *schedulers: torch.optim.lr_scheduler._LRScheduler
        | type["AnyScheduler"],
    ):
        """
        Init.

        Args:
            *schedulers: schedulers.
        """
        self.schedulers = schedulers

    def step(self):
        """Performs step for both schedulers."""
        [s.step() for s in self.schedulers]
