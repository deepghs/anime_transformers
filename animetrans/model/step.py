import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Union

from PIL import Image
from transformers import AutoModel

from .timm_model import TimmModel


@dataclass
class ModelStep:
    model: TimmModel
    epoch: int
    metrics: Dict[str, Union[str, float, Image.Image]]

    def save_to_dir(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        metrics_to_save = {}
        images_to_save = {}
        for key, value in self.metrics.items():
            if not isinstance(value, Image.Image):
                metrics_to_save[key] = value
            else:
                images_to_save[key] = value
        with open(os.path.join(directory, 'metrics.json'), 'w') as f:
            json.dump({
                'epoch': self.epoch,
                **metrics_to_save,
            }, f, sort_keys=True, ensure_ascii=False, indent=4)
        for key, value in images_to_save.items():
            value.save(os.path.join(directory, f'{key}.png'))

    @classmethod
    def load_from_dir(cls, directory: str) -> 'ModelStep':
        model = AutoModel.from_pretrained(directory, trust_remote_code=True)
        with open(os.path.join(directory, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        epoch = metrics.pop('epoch')
        for png_file in glob.glob(os.path.join(directory, '*.png')):
            key = os.path.splitext(os.path.basename(png_file))[0]
            image = Image.open(png_file)
            image.load()
            metrics[key] = image

        return cls(
            model=model,
            epoch=epoch,
            metrics=metrics,
        )
