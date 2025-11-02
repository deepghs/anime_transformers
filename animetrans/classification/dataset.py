import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

from PIL import Image
from datasets import load_dataset as _timm_load_dataset
from huggingface_hub import hf_hub_download
from imgutils.data import load_image


@dataclass
class ClassesInfo:
    classes: List[str]
    classes_to_id: Dict[str, int]


def load_classes(repo_id: str) -> ClassesInfo:
    with open(hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename='classes.json',
    ), 'r') as f:
        classes = json.load(f)

    return ClassesInfo(
        classes=classes,
        classes_to_id={x: i for i, x in enumerate(classes)}
    )


def load_dataset(repo_id: str, split: str = 'train', class_key: str = 'class', image_key: str = 'webp',
                 transforms: Optional = None,
                 row_level_preprocess: Optional[Callable[[Image.Image, dict], Tuple[Image.Image, dict]]] = None):
    dataset = _timm_load_dataset(repo_id, split=split)
    classes_info = load_classes(repo_id)
    classes_to_id = classes_info.classes_to_id
    row_level_preprocess = row_level_preprocess or (lambda x, y: (x, y))

    def _trans(row):
        if row_level_preprocess is not None:
            for i, (image, json_) in enumerate(zip(row[image_key], row['json'])):
                image, json_ = row_level_preprocess(image, json_)
                row[image_key][i], row['json'][i] = image, json_

        images = []
        for image in row[image_key]:
            image = load_image(image, force_background='white', mode='RGB')
            if transforms:
                image = transforms(image)
            images.append(image)
        row['image'] = images

        all_classes = []
        for json_ in row['json']:
            all_classes.append(classes_to_id[json_[class_key]])
        row['classes'] = all_classes
        return row

    dataset = dataset.with_transform(_trans)
    return dataset
