from abc import ABC, abstractmethod

from torch.utils.data import Dataset, IterableDataset
from legm import LoggingMixin


def _parse_namespace(func):
    def wrapper(*args, **kwargs):
        ns = kwargs.get("init__namespace", None)
        if ns:
            if not hasattr(ns, "__len__"):
                ns = [ns]
            for n in ns:
                kwargs.update(
                    {k: getattr(n, k) for k in dir(n) if k not in kwargs}
                )
        return func(*args, **kwargs)

    return wrapper


class BaseDataset(LoggingMixin, Dataset, ABC):
    @_parse_namespace
    def __new__(cls, *args, **kwargs):
        """If argument "splits" (if not provided as kwarg,
        then assumed to be the second arguments) is empty,
        returns None."""
        # assume splits always second argument
        if "splits" in kwargs:
            splits = kwargs["splits"]
        else:
            try:
                splits = args[1]
            except IndexError:
                return super().__new__(cls)
        if not splits:
            return None
        return super().__new__(cls)

    @property
    @abstractmethod
    def name(self):
        return self.__class__.__name__


class BaseIterableDataset(LoggingMixin, IterableDataset, ABC):
    @_parse_namespace
    def __new__(cls, *args, **kwargs):
        """If argument "splits" (if not provided as kwarg,
        then assumed to be the second arguments) is empty,
        returns None."""
        # assume splits always second argument
        if "splits" in kwargs:
            splits = kwargs["splits"]
        else:
            try:
                splits = args[1]
            except IndexError:
                return super().__new__(cls)
        if not splits:
            return None
        return super().__new__()

    @property
    @abstractmethod
    def name(self):
        return self.__class__.__name__
