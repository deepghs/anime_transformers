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
        with open(os.path.join(directory, 'metrics.json'), 'w') as f:
            json.dump({
                'epoch': self.epoch,
                **self.metrics,
            }, f, sort_keys=True, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_dir(cls, directory: str) -> 'ModelStep':
        model = AutoModel.from_pretrained(directory, trust_remote_code=True)
        with open(os.path.join(directory, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        epoch = metrics.pop('epoch')
        return cls(
            model=model,
            epoch=epoch,
            metrics=metrics,
        )
