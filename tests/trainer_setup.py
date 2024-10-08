from typing import Mapping, Sequence
import gridparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from legm import ExperimentManager
from legm.argparse_utils import add_arguments, add_metadata

from ember.trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def input_batch_args(self, batch):
        return batch[1]

    def batch_ids(self, batch: Sequence[torch.Any] | Mapping[str, torch.Any]):
        return batch[0].tolist()

    def calculate_regularization_loss(
        self,
        intermediate_representations: torch.Tensor | None,
        logits: torch.Tensor,
        batch: Sequence[torch.Any],
        train: bool,
        aggregate: bool = True,
        epoch: int | None = None,
    ) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        entropy = (-probs.log() * probs).sum(dim=-1)
        if aggregate:
            entropy = entropy.mean()

        l2 = sum(p.square().sum() for p in self.model.parameters())

        loss = {"entropy": -entropy, "l2": l2}
        coef = {"entropy": 0.1, "l2": 0.002}

        return loss, coef


def parse_args():
    parser = gridparse.GridArgumentParser()
    metadata = {}

    add_arguments(parser, Trainer.argparse_args(), replace_underscores=True)
    add_metadata(metadata, Trainer.argparse_args())

    return parser.parse_args()[0], metadata


def main():
    args, metadata = parse_args()
    print(args)

    train_len = 100_000
    dev_len = 20_000
    test_len = 30_000
    feat_dim = 10
    num_classes = 3

    # create dummy datasets and model
    train_ds = TensorDataset(
        torch.arange(train_len),
        torch.randn(train_len, feat_dim),
        torch.randint(0, num_classes, (train_len,)),
    )
    dev_ds = TensorDataset(
        torch.arange(dev_len),
        torch.randn(dev_len, feat_dim),
        torch.randint(0, num_classes, (dev_len,)),
    )
    test_ds = TensorDataset(
        torch.arange(test_len),
        torch.randn(test_len, feat_dim),
        torch.randint(0, num_classes, (test_len,)),
    )
    model = nn.Linear(feat_dim, num_classes)

    exp_mngr = ExperimentManager("logs/", "Tests", logging_level="debug")
    exp_mngr.set_namespace_params(args)
    exp_mngr.set_param_metadata(metadata, args)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        test_dataset=test_ds,
        experiment_manager=exp_mngr,
    )

    trainer.run()


if __name__ == "__main__":
    main()
