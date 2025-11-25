from typing import List, Type, Optional

from PIL import Image
from hbutils.encoding import int_hash


class RowLevelProcessor:
    def on_train(self, image: Image.Image, json_: dict):
        return image, json_

    def on_eval(self, image: Image.Image, json_: dict):
        return image, json_

    def on_test(self, image: Image.Image, json_: dict):
        return self.on_eval(image, json_)

    def change_classes(self, classes: Optional[List[str]]):
        return classes


def get_int_hash(json_: dict):
    return int_hash(str(json_['id']))


_KNOWN_RPROCESSOR = {}


def register_rprocessor(task_type: str, name: str):
    def _decorator(func):
        if task_type not in _KNOWN_RPROCESSOR:
            _KNOWN_RPROCESSOR[task_type] = {}
        _KNOWN_RPROCESSOR[task_type][name] = func
        return func

    return _decorator


def get_rprocessor(task_type: str, name: str) -> Type[RowLevelProcessor]:
    return _KNOWN_RPROCESSOR[task_type][name]
