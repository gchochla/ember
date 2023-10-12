# Ember

Basic utils and base classes for training and evaluating models in PyTorch. Includes:

* `BaseTrainer`
* `BaseDataset`
* `EarlyStopping` with a PyTorch `Scheduler` interface plus other methods.

The base classes are designed with the `argparse_args` functionalities of [LeGM](https://github.com/gchochla/legm) in mind. `BaseTrainer` is also using `ExperimentManager` from [LeGM](https://github.com/gchochla/legm) directly.