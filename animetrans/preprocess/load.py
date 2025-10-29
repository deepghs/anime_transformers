import logging
import os.path
from typing import Union, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
from imgutils.preprocess import parse_torchvision_transforms
from timm import create_model as _timm_create_model
from timm.data import create_transform as _timm_create_transform, resolve_data_config
from transformers import AutoImageProcessor

from .imgutils_preprocessor import ImgutilsBasedImageProcessor

SplitTyping = Union['train', 'val', 'test']


def load_preprocessor_from_timm(model_name, split: SplitTyping = 'val'):
    logging.info(f'Loading preprocessor of model {model_name!r} as split {split!r} with timm format ...')
    timm_model = _timm_create_model(model_name=model_name, pretrained=False)
    config = resolve_data_config({}, model=timm_model, use_test_size=split == 'test')
    trans = _timm_create_transform(**config, is_training=split == 'train')
    return ImgutilsBasedImageProcessor(
        stages=parse_torchvision_transforms(trans)
    )


def load_preprocessor_from_transformers(model_name):
    logging.info(f'Loading preprocessor of model {model_name!r} with transformers format ...')
    return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_preprocessor(model_name: str, split: SplitTyping = 'val', hf_token: Optional[str] = None):
    if os.path.exists(os.path.join(model_name, 'preprocessor_config.json')):
        return load_preprocessor_from_transformers(model_name)
    elif model_name.startswith('hf-hub:'):
        return load_preprocessor_from_timm(model_name, split=split)
    else:
        try:
            hf_hub_download(
                repo_id=model_name,
                repo_type='model',
                filename='preprocessor_config.json',
                token=hf_token,
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return load_preprocessor_from_timm(f'hf-hub:{model_name}', split=split)
        else:
            return load_preprocessor_from_transformers(model_name)
