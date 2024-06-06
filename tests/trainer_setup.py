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
        return batch[0]


def parse_args():
    parser = gridparse.GridArgumentParser()
    metadata = {}

    add_arguments(parser, Trainer.argparse_args(), replace_underscores=True)
    add_metadata(metadata, Trainer.argparse_args())

    return parser.parse_args()[0], metadata


def main():
    args, metadata = parse_args()
    print(args)

    # create dummy datasets and model
    train_ds = TensorDataset(torch.randn(100, 10), torch.randint(0, 3, (100,)))
    dev_ds = TensorDataset(torch.randn(20, 10), torch.randint(0, 3, (20,)))
    test_ds = TensorDataset(torch.randn(30, 10), torch.randint(0, 3, (30,)))
    model = nn.Linear(10, 3)

    exp_mngr = ExperimentManager("logs/", "Tests")
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
