import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Literal, Any

import torch
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset as _timm_load_dataset
from ditk import logging
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from torch.utils.data import DataLoader

from animetrans.preprocess import load_preprocessor


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


class TransformersTrans:
    """
    Wrapper for Transformers library image preprocessors to make them compatible with standard transform pipelines.

    This class adapts Transformers preprocessors (which expect lists of images and return dictionaries)
    to work with single PIL images in standard PyTorch transform pipelines. It extracts the pixel_values
    from the preprocessor output and returns the first (and only) processed image.
    """

    def __init__(self, transform):
        """
        Initialize the TransformersTrans wrapper.

        :param transform: A Transformers library image preprocessor object.
        :type transform: Any
        """
        self.transform = transform

    def __call__(self, image: Image.Image):
        """
        Apply the Transformers preprocessor to a single PIL image.

        :param image: Input PIL image to transform.
        :type image: Image.Image

        :return: Processed image tensor from the preprocessor.
        :rtype: torch.Tensor
        """
        return self.transform([image])['pixel_values'][0]

    def __repr__(self) -> str:
        """
        Return string representation of the transform wrapper.

        :return: String representation with properly formatted transform details.
        :rtype: str
        """
        # Get the transform's repr and handle multi-line formatting
        transform_repr = repr(self.transform)

        # If the transform repr is multi-line, indent each line properly
        if '\n' in transform_repr:
            # Split into lines and indent each line except the first
            lines = transform_repr.split('\n')
            indented_lines = [lines[0]]  # First line without extra indentation
            for line in lines[1:]:
                indented_lines.append('    ' + line)  # Indent subsequent lines
            transform_repr = '\n'.join(indented_lines)

        return f"{self.__class__.__name__}(\n    transform={transform_repr}\n)"


def load_dataloader(repo_id: str, preprocessor, class_key: str = 'class',
                    split: Literal['train', 'test', 'validation'] = 'train', aug_args: Optional[Dict[str, Any]] = None,
                    batch_size: int = 64, num_workers: int = 32, is_main_process: bool = True, image_key: str = 'webp',
                    row_level_preprocess: Optional[Callable[[Image.Image, dict], Tuple[Image.Image, dict]]] = None):
    from .augmentation import create_augmentation
    trans = create_augmentation(**dict(aug_args or {}))
    trans = T.Compose([
        trans.transforms,
        TransformersTrans(preprocessor),
    ])
    if is_main_process:
        logging.info(f'Transforms loaded (for {split}):\n{trans}')

    if is_main_process:
        logging.info(f'Loading dataset from {repo_id!r} (for {split}) ...')
    row_level_preprocess = row_level_preprocess or (lambda x, y: (x, y))
    dataset = load_dataset(
        repo_id=repo_id,
        split=split,
        transforms=trans,
        image_key=image_key,
        class_key=class_key,
        row_level_preprocess=row_level_preprocess,
    )

    def collate_fn(examples):
        images = []
        classes = []
        for example in examples:
            images.append((example["image"]))
            classes.append(example["classes"])

        pixel_values = torch.stack(images)
        classes = torch.as_tensor(classes)
        return pixel_values, classes

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split == 'train',
        drop_last=split == 'train',
    )
    return dataloader


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    ds = load_dataloader(
        repo_id='deepghs/ai-check-10k',
        preprocessor=load_preprocessor('hf-hub:animetimm/mobilenetv4_conv_aa_large.dbv4-full'),
    )
