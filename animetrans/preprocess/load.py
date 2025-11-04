import json
import logging
import os.path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
from imgutils.preprocess import parse_torchvision_transforms
from imgutils.preprocess.transformers import create_transforms_from_transformers
from timm import create_model as _timm_create_model
from timm.data import create_transform as _timm_create_transform, resolve_data_config
from timm.models import parse_model_name
from transformers import AutoImageProcessor

from .imgutils_preprocessor import ImgutilsBasedImageProcessor


def _get_timm_repo_id(timm_model_name: str):
    model_source, model_name = parse_model_name(timm_model_name)
    if model_source == 'hf-hub':
        return model_name
    else:
        return f'timm/{model_name}'


def load_preprocessor_from_timm(
        model_name: str, use_pre_align: Optional[bool] = None, pre_align_size: int = 512,
        size: Optional[int] = None,
):
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
        config = resolve_data_config(timm_model.pretrained_cfg, model=timm_model, use_test_size=True)
        trans = _timm_create_transform(**config, is_training=False)
        trans_config = parse_torchvision_transforms(trans)
    else:
        trans_config = preprocess_json['test']

    # TODO: better to make these part as function, do not duplicate it

    # TODO: refactor this logic
    #       1. if use_pre_align == true and first stage is not pad_to_size, add it
    #       2. if use_pre_align == true and first stage is pad_to_size but not the given pre_align_size, replace it
    #       3. if use_pre_align == false and first stage is not pad_to_size, do nothing
    #       4. if use_pre_align == false and first stage is pad_to_size, remove it
    #       5. if use_pre_align is None, do nothing
    if use_pre_align is not None:
        if use_pre_align and (not trans_config or trans_config[0]['type'] != 'pad_to_size'):
            # need to add pre align
            trans_config = [
                {
                    "background_color": "white",
                    "interpolation": "bilinear",
                    "size": [pre_align_size, pre_align_size],
                    "type": "pad_to_size"
                },
                *trans_config,
            ]
        elif not use_pre_align and trans_config and trans_config[0]['type'] == 'pad_to_size':
            # need to remove pre align
            trans_config = trans_config[1:]

    # TODO: refactor this logic, define the last stage before to_tensor/maybe_to_tensor (when has to_tensor/maybe_to_tensor)
    #       or the last stage (when no to_tensor/maybe_to_tensor found) as 'last key stage'
    #       1. if 'last key stage' is center_crop, replace its size to [size, size]
    #       2. if 'last key stage' is resize, replace its size to size
    #       3. if 'last key stage' is not those, add a resize stage with size after it
    #       4. if no 'last key stage' found, add one at its place
    if size is not None:
        for item in trans_config:
            if item['type'] == 'resize':
                item['size'] = size
            elif item['type'] == 'center_crop':
                item['size'] = [size, size]

    return ImgutilsBasedImageProcessor(stages=trans_config)


def load_preprocessor_from_transformers(
        model_name: str, use_pre_align: Optional[bool] = None, pre_align_size: int = 512,
        size: Optional[int] = None, **kwargs
):
    logging.info(f'Loading preprocessor of model {model_name!r} with transformers format ...')
    trans = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    trans_config = parse_torchvision_transforms(create_transforms_from_transformers(trans))

    if use_pre_align is not None:
        if use_pre_align and (not trans_config or trans_config[0]['type'] != 'pad_to_size'):
            # need to add pre align
            trans_config = [
                {
                    "background_color": "white",
                    "interpolation": "bilinear",
                    "size": [pre_align_size, pre_align_size],
                    "type": "pad_to_size"
                },
                *trans_config,
            ]
        elif not use_pre_align and trans_config and trans_config[0]['type'] == 'pad_to_size':
            # need to remove pre align
            trans_config = trans_config[1:]

    if size is not None:
        for item in trans_config:
            if item['type'] == 'resize':
                item['size'] = size
            elif item['type'] == 'center_crop':
                item['size'] = [size, size]

    return ImgutilsBasedImageProcessor(stages=trans_config)


def load_preprocessor(model_name: str, hf_token: Optional[str] = None, **kwargs):
    if os.path.exists(os.path.join(model_name, 'preprocessor_config.json')):
        return load_preprocessor_from_transformers(model_name, **kwargs)
    elif model_name.startswith('hf-hub:'):
        return load_preprocessor_from_timm(model_name, **kwargs)
    else:
        if '/' not in model_name:
            return load_preprocessor_from_timm(model_name, **kwargs)

        try:
            hf_hub_download(
                repo_id=model_name,
                repo_type='model',
                filename='preprocessor_config.json',
                token=hf_token,
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return load_preprocessor_from_timm(f'hf-hub:{model_name}', **kwargs)
        else:
            return load_preprocessor_from_transformers(model_name, **kwargs)
