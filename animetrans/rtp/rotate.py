import random
from typing import List, Optional

from PIL import Image

from .base import RowLevelProcessor, get_int_hash, register_rprocessor


def rotate_as_degree(image: Image.Image, r: int):
    return image.rotate(-r, expand=False)


_ROTATE_DEGREES = [0, 90, 180, 270]


@register_rprocessor('classification', 'rotate')
class RotateProcessor(RowLevelProcessor):
    def on_eval(self, image: Image.Image, json_: dict):
        degree = _ROTATE_DEGREES[random.randint(0, len(_ROTATE_DEGREES) - 1)]
        if random.randint(0, 1) == 1:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = rotate_as_degree(image, degree)
        json_['class'] = f'r{degree}'
        return image, json_

    def on_train(self, image: Image.Image, json_: dict):
        ih = get_int_hash(json_)
        degree = _ROTATE_DEGREES[ih % len(_ROTATE_DEGREES)]
        if ih % 2 == 1:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = rotate_as_degree(image, degree)
        json_['class'] = f'r{degree}'
        return image, json_

    def change_classes(self, classes: Optional[List[str]]):
        return [f'r{degree}' for degree in _ROTATE_DEGREES]
