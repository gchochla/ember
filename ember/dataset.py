from abc import ABC, abstractmethod

from torch.utils.data import Dataset, IterableDataset


def _parse_namespace(func):
    def wrapper(*args, **kwargs):
        ns = kwargs.get("init__namespace", None)
        if ns:
            if not hasattr(ns, "__len__"):
                ns = [ns]
            for n in ns:
                kwargs.update({k: getattr(n, k) for k in dir(n)})
        return func(*args, **kwargs)

    return wrapper


class BaseDataset(Dataset, ABC):
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


class BaseIterableDataset(IterableDataset):
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
