import json
import logging
import os.path
from typing import Union, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
from imgutils.preprocess import parse_torchvision_transforms, create_torchvision_transforms
from timm import create_model as _timm_create_model
from timm.data import create_transform as _timm_create_transform, resolve_data_config
from timm.models import parse_model_name
from transformers import AutoImageProcessor

from .imgutils_preprocessor import ImgutilsBasedImageProcessor

SplitTyping = Union['train', 'val', 'test']


def _get_timm_repo_id(timm_model_name: str):
    model_source, model_name = parse_model_name(timm_model_name)
    if model_source == 'hf-hub':
        return model_name
    else:
        return f'timm/{model_name}'


def load_preprocessor_from_timm(model_name: str):
    logging.info(f'Loading preprocessor of model {model_name!r} as timm format ...')
    repo_id = _get_timm_repo_id(model_name)

    try:
        with open(hf_hub_download(
                repo_id=repo_id,
                repo_type='model',
                filename='preprocess.json'
        ), 'r') as f:
            preprocess_json = json.load(f)
    except (RepositoryNotFoundError, EntryNotFoundError):
        timm_model = _timm_create_model(model_name=model_name, pretrained=False)
        config = resolve_data_config({}, model=timm_model, use_test_size=True)
        trans = _timm_create_transform(**config, is_training=False)
    else:
        trans = create_torchvision_transforms(preprocess_json['test'])
    return ImgutilsBasedImageProcessor(
        stages=parse_torchvision_transforms(trans)
    )


def load_preprocessor_from_transformers(model_name: str):
    logging.info(f'Loading preprocessor of model {model_name!r} with transformers format ...')
    return AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_preprocessor(model_name: str, hf_token: Optional[str] = None):
    if os.path.exists(os.path.join(model_name, 'preprocessor_config.json')):
        return load_preprocessor_from_transformers(model_name)
    elif model_name.startswith('hf-hub:'):
        return load_preprocessor_from_timm(model_name)
    else:
        if '/' not in model_name:
            return load_preprocessor_from_timm(model_name)

        try:
            hf_hub_download(
                repo_id=model_name,
                repo_type='model',
                filename='preprocessor_config.json',
                token=hf_token,
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return load_preprocessor_from_timm(f'hf-hub:{model_name}')
        else:
            return load_preprocessor_from_transformers(model_name)
