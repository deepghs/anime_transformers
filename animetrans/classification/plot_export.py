"""
Matplotlib plotting utilities for converting plots to PIL Images.

This module provides utilities for exporting matplotlib plots as PIL Image objects,
with support for converting PyTorch tensors and numpy arrays to appropriate formats.
The main functionality includes tensor/array conversion and plot export capabilities.
"""

import os.path
from typing import Tuple, Any

import numpy as np
import torch
from PIL import Image
from hbutils.system import TemporaryDirectory
from matplotlib import pyplot as plt


def _to_numpy(x: Any) -> Any:
    """
    Convert input data to numpy format recursively.

    This function handles conversion of PyTorch tensors to numpy arrays and
    recursively processes nested data structures like dictionaries, lists, and tuples.
    Non-tensor/array data is returned unchanged.

    :param x: Input data to convert. Can be a tensor, array, dict, list, tuple, or other type.
    :type x: Any

    :return: Converted data with tensors converted to numpy arrays.
    :rtype: Any
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, dict):
        return type(x)({key: _to_numpy(value) for key, value in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([_to_numpy(item) for item in x])
    else:
        return x


def plt_export(func, *args, figsize: Tuple[int, int] = (6, 6), **kwargs) -> Image.Image:
    """
    Export a matplotlib plot function as a PIL Image.

    This function creates a matplotlib figure, calls the provided plotting function
    with the figure's axes, and exports the result as a PIL Image. All input arguments
    are automatically converted from PyTorch tensors to numpy arrays if needed.

    :param func: The plotting function to call. Should accept an axes object as first parameter.
    :type func: callable
    :param args: Positional arguments to pass to the plotting function (after axes).
    :type args: tuple
    :param figsize: Size of the figure in inches (width, height).
    :type figsize: Tuple[int, int]
    :param kwargs: Keyword arguments to pass to the plotting function.
    :type kwargs: dict

    :return: The exported plot as a PIL Image in RGB format.
    :rtype: Image.Image
    """
    fig = plt.Figure(figsize=figsize)
    fig.tight_layout()
    func(fig.gca(), *_to_numpy(args), **_to_numpy(kwargs))

    with TemporaryDirectory() as td:
        imgfile = os.path.join(td, 'image.png')
        fig.savefig(imgfile)

        image = Image.open(imgfile)
        image.load()
        image = image.convert('RGB')
        return image
