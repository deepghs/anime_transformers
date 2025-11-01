from typing import Mapping, Union

from PIL import Image

from ..model import TimmModel


class BaseLogger:
    def __init__(self, workdir, **kwargs):
        _ = kwargs
        self.workdir = workdir

    def train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError

    def eval_log(self, global_step, model: TimmModel, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError
