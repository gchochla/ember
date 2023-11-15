import logging
from copy import deepcopy

import torch


class LoggingMixin:
    """Mixin for logging.

    Initializes a logger if `logging_file` is provided
    and provides a `log` method.

    Attributes:
        _logging_mixin_data: dict for the logging mixin.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        logging_file: str | None = None,
        logging_level: int | str | None = None,
        logger_name: str = None,
        *args,
        **kwargs,
    ):
        """Initializes the logging mixin.

        Args:
            logger: logger to use. Priority over `logging_file`.
            logging_file: logging file.
            logging_level: logging level.
            logger_name: name of the logger.
        """
        self._logging_mixin_data = {}

        if logger is not None:
            self._logging_mixin_data["logger"] = logger
        elif logging_file is not None:
            self.create_logger(logging_file, logging_level, logger_name)

        super().__init__(*args, **kwargs)

    def create_logger(
        self,
        logging_file,
        logging_level: int | str | None = None,
        name: str | None = None,
    ):
        """Creates a logger.

        Args:
            logging_file: logging file.
            logging_level: logging level.
            name: name of the logger.
        """

        if name is None:
            name = self.__class__.__name__

        self._logging_mixin_data["logger"] = logging.getLogger(name)

        if logging_level is None:
            logging_level = logging.WARNING
        elif isinstance(logging_level, str):
            logging_level = getattr(logging, logging_level.upper())
        self._logging_mixin_data["logger"].setLevel(logging_level)

        fh = logging.FileHandler(logging_file)
        fh.setLevel(logging_level)
        formatter = logging.Formatter(
            "%(levelname)s-%(name)s(%(asctime)s)   %(message)s"
        )
        fh.setFormatter(formatter)

        self._logging_mixin_data["logger"].addHandler(fh)

    def get_logger(self) -> logging.Logger:
        """Returns the logger."""
        return self._logging_mixin_data["logger"]

    def log(self, message: str, level: int | str | None = None):
        """Logs a message.

        Args:
            message: message to log.
            level: logging level.
        """
        if level is None:
            level = logging.WARNING
        elif isinstance(level, str):
            level = getattr(logging, level.upper())

        self._logging_mixin_data["logger"].log(level, message)


def set_parameter_requires_grad(
    model: torch.nn.Module, requires_grad: bool = False
):
    """Sets requires_grad for all the parameters in a model (in-place).

    Args:
        model: model to alter.
        requires_grad: whether the model requires grad.
    """
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def flatten_list(l: list, order: int | None = None):
    """Flattens a list up to `order-1` times.

    Args:
        l: the list in question
        order: the depth of the current list,
            `None` if depth is the same for all elements
            (ergo can be discovered automatically)

    Returns:
        A list that has been flattened `order-1` times.
    """

    if not isinstance(l, list):
        l = list(l)

    if order is None:
        lc = deepcopy(l)
        order = 0
        while isinstance(lc, list) and lc:
            lc = lc[0]
            order += 1
    if order == 1:
        return l
    return [lll for ll in l for lll in flatten_list(ll, order - 1)]
