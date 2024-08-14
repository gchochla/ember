import os
from typing import Sequence, Any, Mapping
from abc import ABC, abstractmethod
from time import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from accelerate import Accelerator

# from dataclasses import dataclass
# from transformers.utils.generic import ModelOutput

from ember.train_utils import EarlyStopping
from legm import ExperimentManager, LoggingMixin


def result_str(results: dict[str, float]):
    return ", ".join(
        [
            (
                f"{key}={value:.4f}"
                if isinstance(value, float)
                else f"{key}={value}"
            )
            for key, value in results.items()
        ]
    )


def identify(idict):
    ids = idict.get("ids", None)

    if ids is None:
        return None

    odict = {}
    for k, vs in idict.items():
        if k == "ids":
            continue
        if vs is None:
            vs = [None] * len(ids)
        for id, v in zip(ids, vs):
            odict.setdefault(id, {}).setdefault(k, v)
    return odict


# class FunctionOutput(ModelOutput):
#     """Extandable dict/namespace that is compatible with
#     torch.nn.parallel.DistributedDataParallel"""

#     def __setattr__(self, name, value):
#         v = super().__setattr__(name, value)
#         super().__setitem__(name, value)
#         return v


# class FunctionOutputWithID(FunctionOutput):
#     """Extandable dict/namespace that is compatible with
#     torch.nn.parallel.DistributedDataParallel. Used for indexed metrics."""

#     ids: list[Any]

#     def identify(self):
#         ids = self.ids

#         if ids is None:
#             return None

#         odict = {}
#         for k, vs in self.items():
#             if k == "ids":
#                 continue
#             if vs is None:
#                 vs = [None] * len(ids)
#             for id, v in zip(ids, vs):
#                 odict.setdefault(id, {}).setdefault(k, v)
#         return odict


# @dataclass
# class EvalOutputID(FunctionOutputWithID):
#     preds: list[int] | None = None
#     scores: list[float] | None = None
#     gt: list[int] | None = None
#     log_outs: list[Any] | None = None
#     classification_losses: list[float] | None = None
#     regularization_losses: list[float] | None = None


# @dataclass
# class EvalOutput(FunctionOutput):
#     classification_loss: float | None = None
#     regularization_loss: float | None = None


# @dataclass
# class Metrics(FunctionOutput):
#     eval_classification_loss: float | None = None
#     eval_regularization_loss: float | None = None


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

        # NOTE: no LoggingMixin.argparse_args() because
        # already in ExperimentManager.argparse_args()

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
                    searchable=True,
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
            accelerate=dict(
                action="store_true",
                help="whether to use accelerate for distributed training",
                metadata=dict(disable_comparison=True),
            ),
            lr=dict(
                default=2e-5, type=float, help="learning rate", searchable=True
            ),
            adam_beta1=dict(
                default=0.9, type=float, help="Adam's beta_1", searchable=True
            ),
            adam_beta2=dict(
                default=0.999, type=float, help="Adam's beta_2", searchable=True
            ),
            adam_epsilon=dict(
                default=1e-8, type=float, help="Adam's epsilon", searchable=True
            ),
            weight_decay=dict(
                default=0,
                type=float,
                help="weight decay to apply (if not zero) to all layers "
                "except all bias and LayerNorm weights in AdamW optimizer.",
                searchable=True,
            ),
            train_batch_size=dict(
                default=32, type=int, help="train batch size", searchable=True
            ),
            eval_batch_size=dict(
                default="args.train_batch_size",  # NOTE: this only works with gridparse
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
            max_steps=dict(type=int, help="max number of steps"),
            num_train_epochs=dict(
                type=int, help="number of training epochs", searchable=True
            ),
            warmup_ratio=dict(
                default=0.1,
                type=float,
                help="ratio of training steps (not epochs)"
                " to warmup lr before linear decay",
                searchable=True,
            ),
            max_grad_norm=dict(
                type=float,
                help="maximum gradient norm for gradient clipping",
                searchable=True,
            ),
            disable_intermediate_checkpoints=dict(
                action="store_true",
                help="disable intermediate checkpoints",
                metadata=dict(disable_comparison=True),
            ),
            disable_indexed_logging=dict(
                action="store_true",
                help="disable indexed logging, aka per-example metrics, for the dev set "
                "(usually to save disk space when examples and metrics are many)",
                metadata=dict(disable_comparison=True),
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

        self.exp_manager = experiment_manager
        self.accelerator = Accelerator(cpu=not self.exp_manager.accelerate)
        self.exp_manager.set_main_process(
            self.accelerator.is_local_main_process
        )
        self.exp_manager.start()
        self.accelerator.wait_for_everyone()
        self.exp_manager.device = self.accelerator.device

        super().__init__(
            *args,
            **kwargs,
            logging_file=self.exp_manager.logging_file,
            logging_level=self.exp_manager.logging_level,
        )

        self.set_main_process(self.accelerator.is_local_main_process)

        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.do_train = train_dataset is not None
        self.do_eval = dev_dataset is not None

        train_name = getattr(train_dataset, "name", "Train")
        dev_name = getattr(dev_dataset, "name", "Dev")
        test_name = getattr(test_dataset, "name", "Test")

        self.eval_dataset_names = (
            train_name + ((" -> " + dev_name) if self.do_eval else "")
            if self.do_train
            else None
        )
        self.do_test = test_dataset is not None
        self.test_dataset_names = (
            train_name + ((" -> " + test_name) if self.do_test else "")
            if self.do_train
            else test_name if self.do_test else None
        )
        self.any_dataset = (
            self.train_dataset or self.dev_dataset or self.test_dataset
        )

        if self.exp_manager.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                self.model,
                self.exp_manager.early_stopping_patience,
                lower_better=self.exp_manager.early_stopping_lower_better,
                get_sd_func=self.get_model_state_dict,
                load_sd_func=self.load_model_state_dict,
                logger=self.get_logger(),
            )
        else:
            self.early_stopping = EarlyStopping(
                self.model,
                None,
                get_sd_func=self.get_model_state_dict,
                load_sd_func=self.load_model_state_dict,
                logger=self.get_logger(),
            )
        if (
            self.exp_manager.early_stopping_metric is not None
            and not self.exp_manager.early_stopping_metric.startswith("dev_")
        ):
            self.exp_manager.early_stopping_metric = (
                "dev_" + self.exp_manager.early_stopping_metric
            )

        self.verbose = not self.exp_manager.disable_tqdm

        self.set_num_steps = (
            self.exp_manager.num_train_epochs is not None
            or self.exp_manager.max_steps is not None
        )

        assert not self.do_train or (
            self.set_num_steps
            or (self.early_stopping.patience is not None and self.do_eval)
        )

        if self.do_train:
            dummy_dl = self.init_dataloader(self.train_dataset, train=True)
            optimizer, scheduler = self.init_optimizer_scheduler(len(dummy_dl))
            if self.exp_manager.eval_steps is None:
                self.exp_manager.eval_steps = len(dummy_dl)
        else:
            optimizer, scheduler = None, None

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log(f"Trainer set up.", "debug")
        for ds, split in zip(
            [self.train_dataset, self.dev_dataset, self.test_dataset],
            ["train", "dev", "test"],
        ):
            if ds is None:
                continue

            if hasattr(ds, "debug_message"):
                self.log(split + ": " + ds.debug_message(), "debug")
            else:
                self.log(f"Example sample from {split}: " + str(ds[0]), "debug")

    @staticmethod
    def get_model_state_dict(model: nn.Module) -> dict[str, nn.Parameter]:
        return model.state_dict()

    @staticmethod
    def load_model_state_dict(
        model: nn.Module, state_dict: dict[str, nn.Parameter]
    ):
        model.load_state_dict(state_dict)

    def _checkpoint_fn(self):
        return os.path.join(
            self.exp_manager._experiment_folder, "intermediate_checkpoint.pt"
        )

    def _checkpoint_dict(self, current_epoch):
        self.accelerator.wait_for_everyone()
        return dict(
            optimizer=getattr(self.optimizer, "state_dict", lambda: None)(),
            scheduler=getattr(self.scheduler, "state_dict", lambda: None)(),
            early_stopping=self.early_stopping.state_dict(),
            model=(
                self.get_model_state_dict(self.model) if self.do_train else None
            ),
            exp_manager=self.exp_manager.__getstate__(),
            current_epoch=current_epoch + 1,
        )

    def need_to_load_intermediate(self):
        """Whether to load intermediate checkpoint."""
        return (
            os.path.exists(self._checkpoint_fn())
            and not self.exp_manager.disable_intermediate_checkpoints
        )

    def save_trainer_checkpoint(self, current_epoch: int):
        """Saves checkpoint for trainer."""
        if not self.exp_manager.disable_intermediate_checkpoints:
            try:
                self.accelerator.save(
                    self._checkpoint_dict(current_epoch), self._checkpoint_fn()
                )
            except KeyboardInterrupt as e:
                self.save_trainer_checkpoint(current_epoch)
                raise e

    def load_trainer_checkpoint(self) -> int:
        """Loads checkpoint for trainer. Returns starting epoch."""
        ckpt_fn = self._checkpoint_fn()
        if os.path.exists(ckpt_fn):
            ckpt = torch.load(ckpt_fn)
            if ckpt["optimizer"]:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt["scheduler"]:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.load_model_state_dict(self.model, ckpt["model"])
            self.early_stopping.load_state_dict(
                ckpt["early_stopping"], self.model
            )
            self.exp_manager.__setstate__(ckpt["exp_manager"])
            self.exp_manager.start()
            self.accelerator.wait_for_everyone()

            return ckpt["current_epoch"]
        else:
            return 0

    def train_init(self):
        warnings.warn(
            "`train_init` is deprecated, use `run_init` instead.",
            DeprecationWarning,
        )
        return self.run_init()

    def run_init(self) -> int:
        """Used when Trainer starts. Returns current epoch
        (could be >0 because of intermediate checkpoint)."""

        if self.need_to_load_intermediate():
            self.model = self.accelerator.prepare(self.model)
            current_epoch = self.load_trainer_checkpoint()
        else:
            current_epoch = 0
            if self.exp_manager.model_load_filename is not None:
                loaded_state_dict = torch.load(
                    self.exp_manager.model_load_filename,
                    map_location=self.exp_manager.device,
                )
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

                self.load_model_state_dict(self.model, loaded_state_dict)

            self.model = self.accelerator.prepare(self.model)

        return current_epoch

    def _save_best_model(self):
        """Loads best model to `model` attribute
        and saves to experiment folder."""

        self.load_model_state_dict(
            self.model, self.early_stopping.best_model_state_dict()
        )
        if self.exp_manager.save_model:
            self.accelerator.wait_for_everyone()

            model_fn = self.exp_manager.get_save_filename()
            torch.save(self.get_model_state_dict(self.model), model_fn)
            if self.exp_manager.model_save_filename:
                os.symlink(model_fn, self.exp_manager.model_save_filename)
            self.model.to(self.exp_manager.device)
            self.log(f"Saved model to {model_fn}", "info")

    def train_end(self):
        warnings.warn(
            "`train_end` is deprecated, use `run_end` instead.",
            DeprecationWarning,
        )
        return self.run_end()

    def run_end(self):
        """Used when training (and evaluation) ends."""
        self.exp_manager.log_metrics()
        # TODO: if something fails after here,
        # exp_manager will log the same experiment twice
        self._save_best_model()
        self.exp_manager.aggregate_results()
        self.exp_manager.plot()
        if os.path.exists(self._checkpoint_fn()):
            self.log("Removing intermediate checkpoint", "debug")
            os.remove(self._checkpoint_fn())

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
                            (
                                v.to(self.exp_manager.device)
                                if torch.is_tensor(v)
                                else v
                            )
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
                        (
                            v2.to(self.exp_manager.device)
                            if torch.is_tensor(v2)
                            else v2
                        )
                        for v2 in v
                    ]
                else:
                    device_batch[k] = v

        return device_batch

    @abstractmethod
    def input_batch_args(
        self, batch: Sequence[Any] | Mapping[str, Any]
    ) -> (
        dict[str, tuple[Any, ...] | dict[str, Any]]
        | dict[str, Any]
        | tuple[Any, ...]
        | Any
    ):
        """Creates args and kwargs dicts from batch for the model."""

    def _get_return_vals_from_model(self, batch):
        """Gets return values from the model."""
        input = self.input_batch_args(batch)
        if isinstance(input, Mapping) and (
            "args" in input or "kwargs" in input
        ):
            args = input.get("args", ())
            kwargs = input.get("kwargs", {})
        elif isinstance(input, Mapping):
            kwargs, args = input, ()
        else:
            args, kwargs = input, {}
            if not isinstance(args, (list, tuple)):
                args = (args,)

        return self.model(*args, **kwargs)

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
        epoch: int | None = None,
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

    def get_log_outputs_from_model(self, return_vals: Any) -> dict[str, Any]:
        """Grabs outputs of the model for logging purposes.
        Outputs should be one per example.

        Args:
            return_vals: return values from the model's forward function.

        Returns:
            Some outputs of the model for logging purposes.
        """

        warnings.warn(
            "`get_log_outputs_from_model` is deprecated, use `get_extra_data_from_model` instead.",
            DeprecationWarning,
        )
        return self.get_extra_data_from_model(return_vals)

    def get_extra_data_from_model(
        self, return_vals: Any, batch: Sequence[Any] | None = None
    ) -> dict[str, list[Any]]:
        """Grabs extra data from the model for logging purposes.
        This could include, e.g., attention maps.

        Args:
            return_vals: return values from the model's forward function.

        Returns:
            Some extra data from the model for logging purposes.
        """

    def calculate_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
        aggregate: bool = True,
        epoch: int | None = None,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, float]]
    ):
        """Calculates train loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.
            aggregate: whether to aggregate loss across batch.

        Returns:
            Loss.
        """
        criterion = nn.CrossEntropyLoss(
            reduction="mean" if aggregate else "none"
        )
        return criterion(logits, labels), 1.0

    def calculate_regularization_loss(
        self,
        intermediate_representations: torch.Tensor | None,
        logits: torch.Tensor,
        batch: Sequence[Any],
        train: bool,
        aggregate: bool = True,
        epoch: int | None = None,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, float]]
    ):
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
            return torch.tensor(0.0, device=self.exp_manager.device), 1.0
        labels = self.batch_labels(batch)
        if not train:
            labels = self.accelerator.gather_for_metrics(labels)
        return torch.zeros(len(labels), device=self.exp_manager.device), 1.0

    def calculate_loss(
        self,
        logits: torch.Tensor,
        batch: Sequence[Any],
        train: bool,
        intermediate_representations: torch.Tensor | None = None,
        aggregate: bool = True,
        epoch: int | None = None,
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

        def unpack_return_loss(loss, default_key):
            if isinstance(loss, (list, tuple)):
                loss, coef = loss
            else:
                if isinstance(loss, Mapping):
                    coef = {k: 1.0 for k in loss}
                else:
                    coef = 1.0

            if not isinstance(loss, Mapping):
                loss = {default_key: loss}
                coef = {default_key: coef}

            return loss, coef

        labels = self.batch_labels(batch)
        if not train:
            labels = self.accelerator.gather_for_metrics(labels)

        train_loss = self.calculate_cls_loss(
            logits, labels, train, aggregate, epoch
        )
        train_loss, train_coef = unpack_return_loss(
            train_loss, "classification_loss"
        )

        reg_loss = self.calculate_regularization_loss(
            intermediate_representations, logits, batch, train, aggregate, epoch
        )
        reg_loss, reg_coef = unpack_return_loss(reg_loss, "regularization_loss")

        total_loss = sum(
            (train_coef | reg_coef)[k] * torch.nan_to_num(loss)
            for k, loss in (train_loss | reg_loss).items()
        )

        return (
            total_loss,
            train_loss,
            reg_loss,
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

        return self.accelerator.prepare(optimizer, scheduler)

    def pre_step_actions(self, step):
        """Actions before update step."""

    def post_step_actions(self, step):
        """Actions after update step."""

    def init_dataloader(self, dataset: Dataset, train: bool, **kwargs):
        """Initializes and returns a DataLoader for the given dataset."""
        return self.accelerator.prepare(
            DataLoader(
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
        )

    def train(self):
        """Trains the model."""

        warnings.warn(
            "`train` is deprecated, use `run` instead.",
            DeprecationWarning,
        )

        return self.run()

    def run(self):
        """(If train set was provided) trains and
        (if a dev set or test set was provided) evaluates the model."""

        self.model = self.model.to(self.exp_manager.device)
        self.model.train()
        current_epoch = self.run_init() or 0

        kwargs = (
            dict(shuffle=True)
            if not isinstance(self.train_dataset, IterableDataset)
            else dict()
        )

        if self.do_train:
            data_loader = self.init_dataloader(
                self.train_dataset, train=True, **kwargs
            )

        if self.do_eval:
            dev_data_loader = self.init_dataloader(
                self.dev_dataset, train=False
            )

        if self.do_test:
            test_data_loader = self.init_dataloader(
                self.test_dataset, train=False
            )

        if self.do_train:
            num_epochs = int(self.exp_manager.num_train_epochs)
            early_stop = False
            n_samples = 0

            for epoch in range(current_epoch, num_epochs):
                if early_stop:
                    # the other early stopping checks only break from inner loop
                    self.load_model_state_dict(
                        self.model,
                        self.early_stopping.best_model_state_dict(
                            cleanup=False
                        ),
                    )
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
                    and self.is_main_process()
                    else data_loader
                )

                for step, batch in enumerate(batch_itr):
                    step += epoch * len(data_loader)

                    early_stop = (
                        self.exp_manager.max_steps is not None
                        and step >= self.exp_manager.max_steps
                    )
                    if early_stop:
                        self.log("Forcibly stopping training", "info")
                        break

                    if (
                        step % self.exp_manager.eval_steps == 0
                        or "train_loss" not in locals()
                    ):
                        # FIRST step of current collection of evaluation steps
                        # will always init when epoch == 0, step == 0
                        train_loss = {}
                        cum_reg_loss = {}
                        time_start = time()
                        step_samples = 0

                    # ACTUAL train loop

                    # now handled by accelerator
                    # batch = self.batch_to_device(batch)

                    self.pre_step_actions(step)

                    return_vals = self._get_return_vals_from_model(batch)
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
                        epoch=epoch,
                    )

                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    if self.exp_manager.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.exp_manager.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()

                    self.post_step_actions(step)

                    # ACTUAL train loop end

                    train_loss = {
                        k: train_loss.get(k, 0)
                        + loss.item() * self.batch_len(batch)
                        for k, loss in cls_loss.items()
                    }
                    cum_reg_loss = {
                        k: cum_reg_loss.get(k, 0)
                        + loss.item() * self.batch_len(batch)
                        for k, loss in reg_loss.items()
                    }

                    n_samples += self.batch_len(batch)
                    step_samples += self.batch_len(batch)

                    if (step + 1) % self.exp_manager.eval_steps == 0:
                        # LAST step of current collection of evaluation steps

                        train_loss = {
                            k: v / step_samples for k, v in train_loss.items()
                        }
                        cum_reg_loss = {
                            k: v / step_samples for k, v in cum_reg_loss.items()
                        }

                        results = dict(
                            time_per_sample=(time() - time_start)
                            / step_samples,
                            **train_loss,
                            **cum_reg_loss,
                        )

                        self.exp_manager.set_dict_metrics(
                            results, step=n_samples
                        )

                        if self.do_eval:
                            aggr_results, sample_info = self.evaluate(
                                dev_data_loader,
                                f"Evaluating after {step+1} steps (epoch {epoch+1})",
                                epoch=epoch,
                            )
                            if self.exp_manager.disable_indexed_logging:
                                log_results = aggr_results
                            else:
                                log_results = {
                                    **aggr_results,
                                    **(sample_info or {}),
                                }
                            self.exp_manager.set_dict_metrics(
                                log_results, step=n_samples, mode="dev"
                            )
                            results.update(
                                {"dev_" + k: v for k, v in aggr_results.items()}
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

                self.save_trainer_checkpoint(epoch)

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
                higher_better=not self.exp_manager.early_stopping_lower_better,
            )

        if self.do_test:
            # if train with early stopping,
            # best model is loaded when breaking out of train loop
            results, sample_info = self.evaluate(test_data_loader, "Testing")
            self.log(
                f"Testing metrics for {self.test_dataset_names}: "
                + result_str(results),
                "info",
            )
            # check if steps need to be added here
            self.exp_manager.set_dict_metrics(
                {**results, **(sample_info or {})}, mode="test"
            )

        self.run_end()

    def get_evals_from_dataset(
        self,
        data_loader: DataLoader,
        tqdm_message: str | None = "Evaluation",
        epoch: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Evaluates the model on the given dataset.

        Returns:
            Inference results without and with IDs, and any extra data from
            `get_extra_data_from_model`.
        """

        batch_itr = (
            tqdm(data_loader, desc=tqdm_message, dynamic_ncols=True)
            if not self.exp_manager.disable_tqdm
            and not isinstance(data_loader.dataset, IterableDataset)
            and self.is_main_process()
            else data_loader
        )

        eval_preds = []
        eval_scores = []
        eval_true = []
        eval_losses = {}
        eval_reg_losses = {}
        eval_loss = {}
        eval_reg_loss = {}
        eval_ids = []
        eval_extras = {}

        for batch in batch_itr:
            batch = self.batch_to_device(batch)

            with torch.no_grad():
                return_vals = self._get_return_vals_from_model(batch)

            logits = self.get_logits_from_model(return_vals, batch, data_loader)
            inter_reprs = self.get_intermediate_repr_from_model(
                return_vals, batch
            )
            extra = self.get_extra_data_from_model(return_vals, batch)

            if logits is not None:
                logits = self.accelerator.gather_for_metrics(logits)
            if inter_reprs is not None:
                inter_reprs = self.accelerator.gather_for_metrics(inter_reprs)
            if extra is not None:
                extra = self.accelerator.gather_for_metrics(extra)

            _, cls_loss, reg_loss = self.calculate_loss(
                logits,
                batch,
                train=False,
                intermediate_representations=inter_reprs,
                aggregate=False,
                epoch=epoch,
            )

            cls_loss = self.accelerator.gather_for_metrics(cls_loss)
            reg_loss = self.accelerator.gather_for_metrics(reg_loss)

            for k in cls_loss:
                if cls_loss[k].ndim > 0:
                    eval_losses.setdefault(k, []).extend(cls_loss[k].tolist())
                else:
                    eval_loss.setdefault(k, []).append(cls_loss[k].item())
            for k in reg_loss:
                if reg_loss[k].ndim > 0:
                    eval_reg_losses.setdefault(k, []).extend(
                        reg_loss[k].tolist()
                    )
                else:
                    eval_reg_loss.setdefault(k, []).append(reg_loss[k].item())

            eval_preds.extend(self.get_eval_preds_from_batch(logits))
            eval_true.extend(
                self.get_eval_true_from_batch(
                    self.accelerator.gather_for_metrics(
                        self.batch_labels(batch)
                    )
                )
            )

            scores = self.get_eval_scores_from_batch(logits)
            if scores:
                eval_scores.extend(scores)

            if extra:
                for k, v in extra.items():
                    if isinstance(v, list):
                        eval_extras.setdefault(k, []).extend(v)
                    else:
                        eval_extras.setdefault(k, []).append(v)

            ids = self.batch_ids(batch)
            if ids:
                eval_ids.extend(self.accelerator.gather_for_metrics(ids))

        ### compute eval metrics
        eval_loss = {
            k: sum(v) / len(v) for k, v in (eval_losses | eval_loss).items()
        }
        eval_reg_loss = {
            k: sum(v) / len(v)
            for k, v in (eval_reg_losses | eval_reg_loss).items()
        }

        eval_extras_id = {}
        for k in eval_extras:
            if len(eval_extras[k]) == len(eval_ids):
                eval_extras_id[k] = eval_extras[k]

        return (
            dict(
                **eval_loss,
                **eval_reg_loss,
            ),
            dict(
                preds=eval_preds,
                scores=eval_scores,
                gt=eval_true,
                ids=eval_ids,
                **eval_losses,
                **eval_reg_losses,
                **eval_extras_id,
            ),
            eval_extras,
        )

    def evaluate(
        self,
        data_loader: DataLoader,
        tqdm_message: str | None = "Evaluation",
        epoch: int | None = None,
    ) -> tuple[dict[str, float]]:
        """Evaluates model on `data_loader`.

        Args:
            data_loader: dataset to evaluate on.
            tqdm_message: what to print if tqdm is used.

        Returns:
            A dict of metrics and a dict of metrics indexed by
            the ID of each example.
        """

        self.model.eval()
        self.eval_init(data_loader)

        eval_outs, eval_outs_id, eval_extras = self.get_evals_from_dataset(
            data_loader, tqdm_message, epoch
        )
        others = self.evaluation_metrics(
            eval_outs,
            eval_outs_id,
            eval_extras,
            data_loader=data_loader,
        )
        if others:
            eval_outs.update(others)

        self.model.train()
        self.eval_end(data_loader)

        return eval_outs, identify(eval_outs_id)

    def get_eval_preds_from_batch(
        self, logits: torch.Tensor
    ) -> list[int] | list[list[int]] | list[float] | list[list[float]]:
        """Returns predictions in batch based on logits."""
        return logits.argmax(-1).tolist()

    def get_eval_scores_from_batch(
        self, logits: torch.Tensor
    ) -> list[list[float]]:
        """Returns prediction scores in batch based on logits."""
        return logits.softmax(-1).tolist()

    def get_eval_true_from_batch(
        self, labels: torch.Tensor
    ) -> list[int] | list[list[int]] | list[float] | list[list[float]]:
        """Returns list of ground-truth labels."""
        return labels.tolist()

    def evaluation_metrics(
        self,
        eval_outs: dict[str, Any],
        eval_outs_id: dict[str, Any],
        eval_extras: dict[str, Any],
        data_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Computes evaluation metrics (beyond evaluation loss).

        Args:
            eval_outs: dict of outputs from evaluation.
            eval_outs_id: dict of list outputs from evaluation with
                correspondings IDs. Always contains keys "preds",
                "gt", "scores", and "ids".
            eval_extras: dict of any extra outputs from evaluation.
            data_loader: dataset loader.

        Returns:
            A dict of metrics.
        """
        _, _, macro_f1_score, _ = precision_recall_fscore_support(
            eval_outs_id["gt"],
            eval_outs_id["preds"],
            average="macro",
            zero_division=0,
        )

        results = dict(
            eval_accuracy=accuracy_score(
                eval_outs_id["gt"], eval_outs_id["preds"]
            ),
            macro_f1_score=macro_f1_score,
        )

        return results
