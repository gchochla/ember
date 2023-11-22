import os
from typing import Sequence, Any, Mapping
from abc import ABC, abstractmethod
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

from ember.utils import flatten_list
from ember.train_utils import EarlyStopping
from legm import ExperimentManager, LoggingMixin


def result_str(results: dict[str, float]):
    return ", ".join(
        [
            f"{key}={value:.4f}"
            if isinstance(value, float)
            else f"{key}={value}"
            for key, value in results.items()
        ]
    )


class BaseTrainer(LoggingMixin, ABC):
    """Base trainer class.

    Attributes:
        model: the model to train.
        train_dataset: train dataset, if any.
        dev_dataset: dev dataset, if any.
        test_dataset: test dataset, if any.
        any_dataset: contains the first dataset that is provided.
            To be used when an attribute constant across splits
            needs to be accessed.
        do_train: whether train dataset has been set.
        do_eval: whether dev dataset has been set.
        do_test: whether test dataset has been set.
        args: training arguments (from `transformers`).
        verbose: whether to log/use tqdm.
        optimizer: optimizer (defined in `train`)
        scheduler: lr scheduler (defined in `train`)
    """

    # these are expected in ExperimentManager,
    # so they are not in the init signature
    @staticmethod
    def argparse_args():
        """Arguments for argparse. For documentation, check
        https://github.com/gchochla/legm/blob/main/docs/argparse.md"""
        args = dict(
            save_model=dict(
                action="store_true",
                help="whether to save model",
                metadata=dict(disable_comparison=True),
            ),
            disable_tqdm=dict(
                action="store_true",
                help="disable tqdm progress bars",
                metadata=dict(disable_comparison=True),
            ),
            model_load_filename=dict(
                type=str,
                help="local checkpoint to load",
                metadata=dict(
                    name=(lambda x: x),
                    name_transform=(lambda x: os.path.basename(x)),
                ),
            ),
            model_save_filename=dict(
                type=str,
                help="local checkpoint to save "
                "(symbolic link to ExperimentHandler saved params)",
                metadata=dict(disable_comparison=True),
            ),
            discard_classifier=dict(
                action="store_true",
                help="whether to discard classifier of incoming weights",
            ),
            classifier_layer_name=dict(
                default="classifier",
                type=str,
                help="how classifier layers are names in Pytorch model,"
                ' default is "classifier"',
            ),
            device=dict(
                default="cpu",
                type=str,
                help="which device to use",
                metadata=dict(disable_comparison=True),
            ),
            lr=dict(default=2e-5, type=float, help="learning rate"),
            adam_beta1=dict(default=0.9, type=float, help="Adam's beta_1"),
            adam_beta2=dict(default=0.999, type=float, help="Adam's beta_2"),
            adam_epsilon=dict(default=1e-8, type=float, help="Adam's epsilon"),
            weight_decay=dict(
                default=0,
                type=float,
                help="weight decay to apply (if not zero) to all layers "
                "except all bias and LayerNorm weights in AdamW optimizer.",
            ),
            train_batch_size=dict(
                default=32, type=int, help="train batch size"
            ),
            eval_batch_size=dict(
                default=32,
                type=int,
                help="eval batch size",
                metadata=dict(disable_comparison=True),
            ),
            dataloader_num_workers=dict(
                default=0,
                type=int,
                help="maximum number of workers for dataloaders",
                metadata=dict(disable_comparison=True),
            ),
            eval_steps=dict(
                type=int,
                help="per how many steps to evaluate on dev, default is epoch",
            ),
            max_steps=dict(default=-1, type=int, help="max number of steps"),
            num_train_epochs=dict(type=int, help="number of training epochs"),
            warmup_ratio=dict(
                default=0.1,
                type=float,
                help="ratio of training steps (not epochs)"
                " to warmup lr before linear decay",
            ),
        )

        early_stopping = EarlyStopping.argparse_args()
        early_stopping["early_stopping_metric"]["default"] = "eval_accuracy"
        args.update(early_stopping)

        return args

    def __init__(
        self,
        model: nn.Module,
        experiment_manager: ExperimentManager,
        train_dataset: Dataset | None = None,
        dev_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            model: model to train.
            experiment_manager: hyperparameter and logging handler.
            train_dataset: train dataset.
            dev_dataset: dev dataset.
            test_dataset: test dataset.
            kwargs: logging related arguments.
        """

        super().__init__(*args, **kwargs)

        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.do_train = train_dataset is not None
        self.do_eval = dev_dataset is not None
        self.eval_dataset_names = (
            train_dataset.name
            + ((" -> " + dev_dataset.name) if self.do_eval else "")
            if self.do_train
            else None
        )
        self.do_test = test_dataset is not None
        self.test_dataset_names = (
            train_dataset.name
            + ((" -> " + test_dataset.name) if self.do_test else "")
            if self.do_train
            else test_dataset.name
            if self.do_test
            else None
        )
        self.any_dataset = (
            self.train_dataset or self.dev_dataset or self.test_dataset
        )
        self.exp_manager = experiment_manager
        self.exp_manager.start()

        if hasattr(self.exp_manager, "early_stopping_patience"):
            self.early_stopping = EarlyStopping(
                self.model,
                self.exp_manager.early_stopping_patience,
                self.exp_manager.save_model,
                lower_better=self.exp_manager.lower_better,
                logger=self.get_logger(),
            )
        else:
            self.early_stopping = EarlyStopping(
                self.model,
                None,
                self.exp_manager.save_model,
                logger=self.get_logger(),
            )

        self.verbose = not self.exp_manager.disable_tqdm

        self.set_num_steps = (
            self.exp_manager.num_train_epochs is not None
            or self.exp_manager.max_steps > -1
        )

        assert not self.do_train or (
            self.set_num_steps
            or (self.early_stopping.patience is not None and self.do_eval)
        )

        self.log(f"Trainer set up.", "debug")

    def train_init(self):
        """Used when training starts."""
        if self.exp_manager.model_load_filename is not None:
            loaded_state_dict = torch.load(self.exp_manager.model_load_filename)
            # if classifier layers exist, then they do not need to match
            if any(
                [
                    self.exp_manager.classifier_layer_name in layer
                    for layer in loaded_state_dict
                ]
            ):
                model_state_dict = self.model.state_dict()
                for k in list(loaded_state_dict):
                    if self.exp_manager.classifier_layer_name in k:
                        if self.exp_manager.discard_classifier:
                            if k in model_state_dict:
                                loaded_state_dict[k] = model_state_dict[k]
                            else:
                                loaded_state_dict.pop(k)

                        elif k in model_state_dict and (
                            loaded_state_dict[k].shape
                            != model_state_dict[k].shape
                        ):
                            loaded_state_dict[k] = model_state_dict[k]

            self.model.load_state_dict(loaded_state_dict)

    def _save_best_model(self):
        """Loads best model to `model` attribute
        and saves to experiment folder."""
        self.model = self.early_stopping.best_model()
        if self.exp_manager.save_model:
            model_fn = self.exp_manager.get_save_filename()
            torch.save(self.model.cpu().state_dict(), model_fn)
            os.symlink(model_fn, self.exp_manager.model_save_filename)
            self.model.to(self.exp_manager.device)
            self.log(f"Saved model to {model_fn}", "info")

    def train_end(self):
        """Used when training (and evaluation) ends."""
        self.exp_manager.log_metrics()
        self._save_best_model()
        self.exp_manager.aggregate_results()
        self.exp_manager.plot()

    def eval_init(self, data_loader: DataLoader):
        """Used when evaluation starts.

        Args:
            data_loader: DataLoader the model is going to be evaluated on.
        """

    def eval_end(self, data_loader: DataLoader):
        """Used when evaluation ends.

        Args:
            data_loader: DataLoader the model was evaluated on.
        """

    def batch_to_device(
        self, batch: Sequence[Any] | Mapping[str, Any]
    ) -> Sequence[Any] | Mapping[str, Any]:
        """Get batch as returned by DataLoader to device."""

        if isinstance(batch, Sequence):
            device_batch = []
            for elem in batch:
                # TENSOR
                if torch.is_tensor(elem):
                    device_batch.append(elem.to(self.exp_manager.device))
                # DICTIONARY
                elif isinstance(elem, dict):
                    device_batch.append(
                        {
                            k: (
                                v.to(self.exp_manager.device)
                                if torch.is_tensor(v)
                                else v
                            )
                            for k, v in elem.items()
                        }
                    )
                # LIST, TUPLE, etc.
                elif isinstance(elem, Sequence):
                    device_batch.append(
                        [
                            v.to(self.exp_manager.device)
                            if torch.is_tensor(v)
                            else v
                            for v in elem
                        ]
                    )
                # OTHER (like strings)
                else:
                    device_batch.append(elem)

        else:
            device_batch = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    device_batch[k] = v.to(self.exp_manager.device)
                elif isinstance(v, Mapping):
                    device_batch[k] = {
                        k2: (
                            v2.to(self.exp_manager.device)
                            if torch.is_tensor(v2)
                            else v2
                        )
                        for k2, v2 in v.items()
                    }
                elif isinstance(v, Sequence):
                    device_batch[k] = [
                        v2.to(self.exp_manager.device)
                        if torch.is_tensor(v2)
                        else v2
                        for v2 in v
                    ]
                else:
                    device_batch[k] = v

        return device_batch

    @abstractmethod
    def input_batch_kwargs(
        self, batch: Sequence[Any] | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Creates a kwargs dict from batch for the model."""

    def batch_labels(self, batch: Sequence[Any] | Mapping[str, Any]):
        """Grabs labels from batch."""
        return batch[-1]

    def batch_ids(self, batch: Sequence[Any] | Mapping[str, Any]):
        """Returns some identifier for the examples of the batch."""

    def get_logits_from_model(
        self,
        return_vals: Any,
        batch: Sequence[Any],
        data_loader: DataLoader,
        epoch: int = -1,
    ) -> torch.Tensor:
        """Grabs logits from model's return values. So far, extra arguments
        a "hacky" way to substitute images for computed embeddings when image
        encoder is frozen.

        Args:
            return_vals: return values from the model's forward function.
            batch: the batch the output was produced by.
            data_loader: the DataLoader the batch came from.
            epoch: the current epoch.

        Returns:
            The logits of the model.
        """
        try:
            return return_vals.logits
        except:
            return return_vals

    def batch_len(self, batch: Sequence[Any]) -> int:
        """Batch size."""
        return len(self.batch_labels(batch))

    def get_intermediate_repr_from_model(
        self, return_vals: Any, batch: Sequence[Any]
    ) -> torch.Tensor | None:
        """Grabs intermediate representations of the model from its output,
        if necessary for some regularization loss.

        Args:
            return_vals: return values from the model's forward function.
            batch: the batch inputs came from.

        Returns:
            Some intermediate representation of the model for
                regularization losses, if necessary.
        """

    def get_log_outputs_from_model(self, return_vals: Any) -> list[Any]:
        """Grabs outputs of the model for logging purposes.

        Args:
            return_vals: return values from the model's forward function.

        Returns:
            Some outputs of the model for logging purposes.
        """

    def calculate_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
        aggregate: bool = True,
    ) -> torch.Tensor:
        """Calculates train loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.
            aggregate: whether to aggregate loss across batch.

        Returns:
            Loss.
        """
        criterion = nn.CrossEntropyLoss(reduction="mean" if aggregate else None)
        return criterion(logits, labels)

    def calculate_regularization_loss(
        self,
        intermediate_representations: torch.Tensor | None,
        logits: torch.Tensor,
        batch: Sequence[Any],
        train: bool,
        aggregate: bool = True,
    ) -> torch.Tensor:
        """Calculates regularization loss based on some intermediate
        representation and the batch information (like labels).

        Args:
            intermediate representations: some intermediate representation
                from the network.
            logits: model predictions.
            batch: the batch this representation came from.
            train: whether this is used during training.
            aggregate: whether to aggregate loss across batch,
                if possible.

        Returns:
            Regularization loss (or a dummy 0 tensor on the proper device).
        """
        if aggregate:
            return torch.tensor(0.0, device=self.exp_manager.device)
        return torch.zeros(
            len(self.batch_labels(batch)), device=self.exp_manager.device
        )

    def calculate_loss(
        self,
        logits: torch.Tensor,
        batch: Sequence[Any],
        train: bool,
        intermediate_representations: torch.Tensor | None = None,
        aggregate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.
            intermediate_representations

        Returns:
            Loss, train loss and regularization loss.
        """
        train_loss = self.calculate_cls_loss(
            logits, self.batch_labels(batch), train, aggregate
        )

        regularization_loss = self.calculate_regularization_loss(
            intermediate_representations, logits, batch, train, aggregate
        )

        return (
            train_loss + regularization_loss,
            train_loss,
            regularization_loss,
        )

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initializes and returns optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.exp_manager.lr,
            betas=[self.exp_manager.adam_beta1, self.exp_manager.adam_beta2],
            eps=self.exp_manager.adam_epsilon,
            weight_decay=self.exp_manager.weight_decay,
        )

    def init_optimizer_scheduler(
        self, num_batches: int
    ) -> tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.ChainedScheduler
    ]:
        """Initializes and returns optimizer (based on `init_optimizer`)
        and scheduler.

        Args:
            num_batches: number of batches in an epoch.
        """
        optimizer = self.init_optimizer()

        if self.set_num_steps:
            if self.exp_manager.num_train_epochs:
                num_epochs = int(self.exp_manager.num_train_epochs)
                num_steps = int(num_batches * num_epochs)
            else:
                num_steps = self.exp_manager.max_steps
            warmup_steps = int(self.exp_manager.warmup_ratio * num_steps)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: 1
            )

        return optimizer, scheduler

    def pre_step_actions(self, step):
        """Actions before update step."""

    def post_step_actions(self, step):
        """Actions after update step."""

    def init_dataloader(self, dataset: Dataset, train: bool, **kwargs):
        """Initializes and returns a DataLoader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=(
                self.exp_manager.train_batch_size
                if train
                else self.exp_manager.eval_batch_size
            ),
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=self.exp_manager.dataloader_num_workers,
            **kwargs,
        )

    def train(self):
        """Trains and, if a dev set or test set was provided, evaluates
        the model."""

        self.model = self.model.to(self.exp_manager.device)
        self.model.train()
        self.train_init()

        kwargs = (
            dict(shuffle=True)
            if not isinstance(self.train_dataset, IterableDataset)
            else dict()
        )

        if self.do_train:
            data_loader = self.init_dataloader(
                self.train_dataset, train=True, **kwargs
            )
            self.optimizer, self.scheduler = self.init_optimizer_scheduler(
                len(data_loader)
            )
        else:
            self.optimizer, self.scheduler = None, None

        if self.do_eval:
            dev_data_loader = self.init_dataloader(
                self.dev_dataset, train=False
            )

        if self.do_test:
            test_data_loader = self.init_dataloader(
                self.test_dataset, train=False
            )

        if self.do_train:
            num_epochs = int(self.exp_manager.num_train_epochs or 1)
            early_stop = False
            n_samples = 0

            for epoch in range(num_epochs):
                if early_stop:
                    # the other early stopping check only breaks from inner loop
                    break

                # use tqdm is possible with dataloader
                batch_itr = (
                    tqdm(
                        data_loader,
                        desc=f"Training Epoch {epoch+1}",
                        dynamic_ncols=True,
                    )
                    if not self.exp_manager.disable_tqdm
                    and not isinstance(data_loader.dataset, IterableDataset)
                    else data_loader
                )

                for step, batch in enumerate(batch_itr):
                    step += epoch * len(data_loader)

                    early_stop = (
                        self.exp_manager.max_steps > -1
                        and step >= self.exp_manager.max_steps
                    )
                    if early_stop:
                        self.log("Forcibly stopping training", "info")
                        break

                    if step % self.exp_manager.eval_steps == 0:
                        # FIRST step of current collection of evaluation steps
                        # will always init when epoch == 0, step == 0
                        train_loss = 0.0
                        cum_regularization_loss = 0.0
                        time_start = time()
                        step_samples = 0

                    # ACTUAL train loop

                    batch = self.batch_to_device(batch)

                    self.pre_step_actions(step)

                    return_vals = self.model(**self.input_batch_kwargs(batch))
                    logits = self.get_logits_from_model(
                        return_vals, batch, data_loader, epoch
                    )
                    inter_repr = self.get_intermediate_repr_from_model(
                        return_vals, batch
                    )

                    loss, cls_loss, reg_loss = self.calculate_loss(
                        logits,
                        batch,
                        train=True,
                        intermediate_representations=inter_repr,
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.post_step_actions(step)

                    # ACTUAL train loop end

                    train_loss += cls_loss.item() * self.batch_len(batch)
                    cum_regularization_loss += reg_loss.item() * self.batch_len(
                        batch
                    )
                    n_samples += self.batch_len(batch)
                    step_samples += self.batch_len(batch)

                    if (step + 1) % self.exp_manager.eval_steps == 0:
                        # LAST step of current collection of evaluation steps

                        train_loss /= step_samples
                        cum_regularization_loss /= step_samples

                        results = dict(
                            train_loss=train_loss,
                            regularization_loss=cum_regularization_loss,
                            time_per_sample=(time() - time_start)
                            / step_samples,
                        )

                        self.exp_manager.set_dict_metrics(
                            results, step=n_samples
                        )

                        if self.do_eval:
                            aggr_results, per_example_results = self.evaluate(
                                dev_data_loader,
                                f"Evaluating after {step+1} steps (epoch {epoch+1})",
                            )
                            self.exp_manager.set_dict_metrics(
                                {**aggr_results, **(per_example_results or {})},
                                step=n_samples,
                            )

                        self.log(
                            f"Step {step+1} (epoch {epoch+1}) metrics on "
                            + self.eval_dataset_names
                            + ": "
                            + result_str(results),
                            "info",
                        )

                        early_stop = self.early_stopping.step(
                            results.get(
                                self.exp_manager.early_stopping_metric, None
                            ),
                            **{**results, "epoch": epoch + 1, "step": step + 1},
                        )
                        if early_stop:
                            self.log(
                                "Early stopping at step "
                                f"{step + 1} (epoch {epoch+1})",
                                "info",
                            )
                            break

        early_stopping_metrics = self.early_stopping.get_metrics()
        if early_stopping_metrics is not None:
            self.log(
                f"Best metrics based on {self.exp_manager.early_stopping_metric} "
                f"on {self.eval_dataset_names}: "
                + result_str(early_stopping_metrics),
                "info",
            )
            # check if steps need to be added here
            self.exp_manager.set_best(
                "early_stopping",
                metric=self.exp_manager.early_stopping_metric,
                higher_better=not self.exp_manager.lower_better,
            )

        if self.do_test:
            results, per_example_results = self.evaluate(
                test_data_loader, "Testing"
            )
            self.log(
                f"Testing metrics for {self.test_dataset_names}: "
                + result_str(results),
                "info",
            )
            # check if steps need to be added here
            self.exp_manager.set_dict_metrics(
                {**results, **(per_example_results or {})}, test=True
            )

        self.train_end()

    def get_evals_from_dataset(
        self,
        data_loader: DataLoader,
        tqdm_message: str | None = "Evaluation",
    ) -> tuple[
        list[list[int]],
        list[list[float]],
        list[list[int]],
        list[Any],
        list[Any],
        list[float],
        list[float],
        float,
        float,
    ]:
        """Evaluates the model on the given dataset.

        Returns:
            eval_preds: list of predictions for each example.
            eval_scores: list of scores for each example.
            eval_true: list of true labels for each example.
            eval_ids: list of ids for each example.
            eval_outs: list of model outputs for each example.
            eval_losses: list of losses for each example.
            eval_reg_losses: list of regularization losses for each example.
            eval_loss: Loss on the dataset.
            eval_reg_loss: Regularization loss on the dataset.
        """

        batch_itr = (
            tqdm(data_loader, desc=tqdm_message, dynamic_ncols=True)
            if not self.exp_manager.disable_tqdm
            and not isinstance(data_loader.dataset, IterableDataset)
            else data_loader
        )

        eval_preds = []
        eval_scores = []
        eval_true = []
        eval_losses = []
        eval_reg_losses = []
        eval_ids = []
        eval_outs = []
        eval_loss = 0.0
        eval_reg_loss = 0.0

        for batch in batch_itr:
            batch = self.batch_to_device(batch)

            with torch.no_grad():
                return_vals = self.model(**self.input_batch_kwargs(batch))

            logits = self.get_logits_from_model(return_vals, batch, data_loader)
            inter_reprs = self.get_intermediate_repr_from_model(
                return_vals, batch
            )
            outs = self.get_log_outputs_from_model(return_vals)

            _, cls_loss, reg_loss = self.calculate_loss(
                logits,
                batch,
                train=False,
                intermediate_representations=inter_reprs,
                aggregate=False,
            )

            eval_loss += cls_loss.sum().item()
            eval_reg_loss += reg_loss.sum().item()

            eval_losses.extend(cls_loss.tolist())
            eval_reg_losses.extend(reg_loss.tolist())

            eval_preds.extend(self.get_eval_preds_from_batch(logits))
            eval_true.extend(
                self.get_eval_true_from_batch(self.batch_labels(batch))
            )

            scores = self.get_eval_scores_from_batch(logits)
            if scores:
                eval_scores.extend(scores)

            if outs:
                eval_outs.extend(outs)

            ids = self.batch_ids(batch)
            if ids:
                eval_ids.extend(ids)

        ### compute eval metrics
        eval_loss /= len(data_loader.dataset)
        eval_reg_loss /= len(data_loader.dataset)

        return (
            eval_preds,
            eval_scores,
            eval_true,
            eval_ids,
            eval_outs,
            eval_losses,
            eval_reg_losses,
            eval_loss,
            eval_reg_loss,
        )

    def evaluate(
        self,
        data_loader: DataLoader,
        tqdm_message: str | None = "Evaluation",
    ) -> tuple[dict[str, float]]:
        """Evaluates model on `data_loader`.

        Args:
            data_loader: dataset to evaluate on.
            tqdm_message: what to print if tqdm is used.
        """

        self.model.eval()
        self.eval_init(data_loader)

        (
            eval_preds,
            eval_scores,
            eval_true,
            eval_ids,
            eval_outs,
            eval_losses,
            eval_reg_losses,
            eval_loss,
            eval_reg_loss,
        ) = self.get_evals_from_dataset(data_loader, tqdm_message)

        results = dict(
            eval_loss=eval_loss, eval_regularization_loss=eval_reg_loss
        )
        others = self.evaluation_metrics(
            eval_true,
            eval_preds,
            data_loader=data_loader,
            eval_ids=eval_ids,
            eval_pred_scores=eval_scores,
        )
        if others:
            results.update(others)

        self.model.train()
        self.eval_end(data_loader)

        per_sample_results = None

        if eval_ids is not None:
            per_sample_results = {
                _id: dict(
                    pred=pred,
                    true=true,
                    score=score,
                    out=out,
                    loss=loss,
                    reg_loss=reg_loss,
                )
                for _id, pred, true, score, out, loss, reg_loss in zip(
                    eval_ids,
                    eval_preds or [None] * len(eval_ids),
                    eval_true or [None] * len(eval_ids),
                    eval_scores or [None] * len(eval_ids),
                    eval_outs or [None] * len(eval_ids),
                    eval_losses or [None] * len(eval_ids),
                    eval_reg_losses or [None] * len(eval_ids),
                )
            }

        return results, per_sample_results

    def get_eval_preds_from_batch(
        self, logits: torch.Tensor
    ) -> list[list[int]]:
        """Returns predictions in batch based on logits."""
        return logits.argmax(-1).tolist()

    def get_eval_scores_from_batch(
        self, logits: torch.Tensor
    ) -> list[list[float]]:
        """Returns prediction scores in batch based on logits."""
        return logits.softmax(-1).tolist()

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> list[list[int]]:
        """Returns list of ground-truth labels."""
        return labels.tolist()

    def evaluation_metrics(
        self,
        eval_true: list[list[int]],
        eval_preds: list[list[int]],
        data_loader: DataLoader,
        eval_ids: list[Any] | None = None,
        eval_pred_scores: list[list[float]] | None = None,
    ) -> dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss).

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.
            data_loader: DataLoader where data came from.
            eval_ids: IDs of examples (used for example to group)
            eval_pred_scores: scores (probabilities) of predictions.

        Returns:
            A dict of metrics.
        """
        _, _, macro_f1_score, _ = precision_recall_fscore_support(
            flatten_list(eval_true),
            flatten_list(eval_preds),
            average="macro",
            zero_division=0,
        )

        eval_accuracy = np.mean(
            [
                pred == label
                for preds, labels in zip(eval_preds, eval_true)
                for pred, label in zip(preds, labels)
            ]
        )

        results = dict(
            eval_accuracy=eval_accuracy,
            macro_f1_score=macro_f1_score,
        )

        return results
