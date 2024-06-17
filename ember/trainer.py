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
            max_steps=dict(default=-1, type=int, help="max number of steps"),
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
        self.exp_manager.start()

        super().__init__(
            *args,
            **kwargs,
            logging_file=self.exp_manager.logging_file,
            logging_level=self.exp_manager.logging_level,
        )

        self.accelerator = Accelerator(cpu=not self.exp_manager.accelerate)
        self.exp_manager.set_main_process(
            self.accelerator.is_local_main_process
        )
        self.set_main_process(self.accelerator.is_local_main_process)
        self.exp_manager.device = self.accelerator.device

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

        if hasattr(self.exp_manager, "early_stopping_patience"):
            self.early_stopping = EarlyStopping(
                self.model,
                self.exp_manager.early_stopping_patience,
                self.exp_manager.save_model,
                lower_better=self.exp_manager.early_stopping_lower_better,
                logger=self.get_logger(),
            )
            print(self.early_stopping.get_logger())
        else:
            self.early_stopping = EarlyStopping(
                self.model,
                None,
                self.exp_manager.save_model,
                logger=self.get_logger(),
            )

        if (
            getattr(self.exp_manager, "eval_steps", None) is None
            and self.do_train
        ):
            self.exp_manager.eval_steps = (
                len(train_dataset) + self.exp_manager.train_batch_size - 1
            ) // self.exp_manager.train_batch_size

        self.verbose = not self.exp_manager.disable_tqdm

        self.set_num_steps = (
            self.exp_manager.num_train_epochs is not None
            or self.exp_manager.max_steps > -1
        )

        assert not self.do_train or (
            self.set_num_steps
            or (self.early_stopping.patience is not None and self.do_eval)
        )

        if self.do_train:
            dummy_dl = self.init_dataloader(self.train_dataset, train=True)

            # otherwise
            # num_batches=(
            #     len(self.train_dataset)
            #     + self.exp_manager.train_batch_size
            #     - 1
            # )
            # // self.exp_manager.train_batch_size
            # and we don't know if drop_last=True, etc

            optimizer, scheduler = self.init_optimizer_scheduler(len(dummy_dl))
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

    def _model_state_dict(self):
        state_dict = (
            self.accelerator.unwrap_model(self.model).cpu().state_dict()
        )
        self.model = self.accelerator.prepare(self.model).to(
            self.exp_manager.device
        )
        return state_dict

    def _checkpoint_fn(self):
        return os.path.join(
            self.exp_manager._experiment_folder, "intermediate_checkpoint.pt"
        )

    def _checkpoint_dict(self, current_epoch):
        return dict(
            optimizer=getattr(self.optimizer, "state_dict", lambda: None)(),
            scheduler=getattr(self.scheduler, "state_dict", lambda: None)(),
            early_stopping=self.early_stopping.state_dict(),
            model=self._model_state_dict() if self.do_train else None,
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
            torch.save(
                self._checkpoint_dict(current_epoch), self._checkpoint_fn()
            )

    def load_trainer_checkpoint(self) -> int:
        """Loads checkpoint for trainer. Returns starting epoch."""
        ckpt_fn = self._checkpoint_fn()
        if os.path.exists(ckpt_fn):
            ckpt = torch.load(ckpt_fn, map_location=self.exp_manager.device)
            # TODO: need to init optimizer before
            if ckpt["optimizer"]:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt["scheduler"]:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.model.load_state_dict(ckpt["model"])
            self.early_stopping.load_state_dict(
                ckpt["early_stopping"], self.model
            )
            self.exp_manager.__setstate__(ckpt["exp_manager"])
            self.exp_manager.start()

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
            current_epoch = self.load_trainer_checkpoint()
        else:
            current_epoch = 0
            if self.exp_manager.model_load_filename is not None:
                loaded_state_dict = torch.load(
                    self.exp_manager.model_load_filename
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

                self.model.load_state_dict(loaded_state_dict)

        self.model = self.accelerator.prepare(self.model)
        return current_epoch

    def _save_best_model(self):
        """Loads best model to `model` attribute
        and saves to experiment folder."""

        self.model = self.early_stopping.best_model()
        if self.exp_manager.save_model:
            self.accelerator.wait_for_everyone()

            model_fn = self.exp_manager.get_save_filename()
            torch.save(self._model_state_dict(), model_fn)
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
        criterion = nn.CrossEntropyLoss(
            reduction="mean" if aggregate else "none"
        )
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
            num_epochs = int(self.exp_manager.num_train_epochs or 1)
            early_stop = False
            n_samples = 0

            for epoch in range(current_epoch, num_epochs):
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
                            aggr_results, sample_info = self.evaluate(
                                dev_data_loader,
                                f"Evaluating after {step+1} steps (epoch {epoch+1})",
                            )
                            self.exp_manager.set_dict_metrics(
                                {**aggr_results, **(sample_info or {})},
                                step=n_samples,
                            )
                            results.update(aggr_results)

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
            results, sample_info = self.evaluate(test_data_loader, "Testing")
            self.log(
                f"Testing metrics for {self.test_dataset_names}: "
                + result_str(results),
                "info",
            )
            # check if steps need to be added here
            self.exp_manager.set_dict_metrics(
                {**results, **(sample_info or {})}, test=True
            )

        self.run_end()

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
                return_vals = self._get_return_vals_from_model(batch)

            logits = self.get_logits_from_model(return_vals, batch, data_loader)
            inter_reprs = self.get_intermediate_repr_from_model(
                return_vals, batch
            )
            outs = self.get_log_outputs_from_model(return_vals)

            if logits is not None:
                logits = self.accelerator.gather_for_metrics(logits)
            if inter_reprs is not None:
                inter_reprs = self.accelerator.gather_for_metrics(inter_reprs)
            if outs is not None:
                outs = self.accelerator.gather_for_metrics(outs)

            _, cls_loss, reg_loss = self.calculate_loss(
                logits,
                batch,
                train=False,
                intermediate_representations=inter_reprs,
                aggregate=False,
            )

            cls_loss = self.accelerator.gather_for_metrics(cls_loss)
            reg_loss = self.accelerator.gather_for_metrics(reg_loss)

            eval_loss += cls_loss.sum().item()
            eval_reg_loss += reg_loss.sum().item()

            eval_losses.extend(cls_loss.tolist())
            eval_reg_losses.extend(reg_loss.tolist())

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

            if outs:
                eval_outs.extend(outs)

            ids = self.accelerator.gather_for_metrics(self.batch_ids(batch))
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
            eval_ids,
            eval_true,
            eval_preds,
            eval_scores,
            eval_losses,
            eval_reg_losses,
            data_loader=data_loader,
        )
        if others:
            results.update(others)

        self.model.train()
        self.eval_end(data_loader)

        sample_info = None

        if eval_ids is not None:
            sample_info = {
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

        return results, sample_info

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
        eval_ids: list[Any],
        eval_true: list[int],
        eval_preds: list[int],
        eval_scores: list[float],
        eval_losses: list[float],
        eval_reg_losses: list[float],
        data_loader: DataLoader | None = None,
    ) -> dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss).

        Args:
            eval_ids: IDs of examples (used for example to group)
            eval_true: ground-truth labels.
            eval_preds: predictions.
            eval_scores: prediction scores.
            eval_losses: losses.
            eval_reg_losses: regularization losses.
            data_loader: DataLoader where data came from.

        Returns:
            A dict of metrics.
        """
        _, _, macro_f1_score, _ = precision_recall_fscore_support(
            eval_true, eval_preds, average="macro", zero_division=0
        )

        results = dict(
            eval_accuracy=accuracy_score(eval_true, eval_preds),
            macro_f1_score=macro_f1_score,
        )

        return results
