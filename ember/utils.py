from copy import deepcopy

import torch


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
