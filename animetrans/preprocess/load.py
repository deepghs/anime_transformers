import json
import logging
import os.path
from typing import Optional, List, Dict, Any, Union, Tuple

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


def _size_align(size: Union[int, List[int], Tuple[int, int]]):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple)):
        if len(size) == 1:
            return size[0], size[0]
        else:
            return tuple(size)
    else:
        raise TypeError(f'Unknown size - {size!r}.')


_SIZE_STAGES = {'center_crop', 'resize', 'pad_to_size'}


def _handle_useless_size_operation(trans_config: List[Dict[str, Any]]):
    _trans = []
    _last_size = None
    for i, trans_item in enumerate(trans_config):
        if i > 0:
            if trans_item['type'] in _SIZE_STAGES:  # this stage is size related
                if _last_size is None or _size_align(trans_item['size']) != _last_size:
                    # this stage will actually change the size
                    _trans.append(trans_item)
                    _last_size = _size_align(trans_item['size'])
            else:
                _trans.append(trans_item)
        else:
            _trans.append(trans_item)
            if trans_item['type'] in _SIZE_STAGES:
                _last_size = _size_align(trans_item['size'])
    return _trans


def _handle_pre_align_logic(
        trans_config: List[Dict[str, Any]],
        use_pre_align: Optional[bool],
        pre_align_size: int
) -> List[Dict[str, Any]]:
    """
    Handle pre-align logic based on use_pre_align parameter:
    1. if use_pre_align == true and first stage is not pad_to_size, add it
    2. if use_pre_align == true and first stage is pad_to_size but not the given pre_align_size, replace it
    3. if use_pre_align == false and first stage is not pad_to_size, do nothing
    4. if use_pre_align == false and first stage is pad_to_size, remove it
    5. if use_pre_align is None, do nothing
    """
    if use_pre_align is None:
        return trans_config

    has_pad_to_size_first = trans_config and trans_config[0]['type'] == 'pad_to_size'

    if use_pre_align:
        pad_to_size_config = {
            "background_color": "white",
            "interpolation": "bilinear",
            "size": [pre_align_size, pre_align_size],
            "type": "pad_to_size"
        }

        if not has_pad_to_size_first:
            # Case 1: add pre align
            trans_config = [pad_to_size_config, *trans_config]
        else:
            # Case 2: replace existing pad_to_size with correct size
            current_size = trans_config[0].get('size', [])
            if current_size != [pre_align_size, pre_align_size]:
                trans_config[0] = pad_to_size_config
    else:  # use_pre_align == False
        if has_pad_to_size_first:
            # Case 4: remove pre align
            trans_config = trans_config[1:]
        # Case 3: do nothing if first stage is not pad_to_size

    return trans_config


def _find_to_tensor_index(trans_config: List[Dict[str, Any]]) -> Optional[int]:
    """
    Find the last key stage before to_tensor/maybe_to_tensor (when has to_tensor/maybe_to_tensor)
    or the last stage (when no to_tensor/maybe_to_tensor found) as 'last key stage'
    """
    tensor_types = {'to_tensor', 'maybe_to_tensor'}

    # Find first tensor conversion stage
    tensor_index = None
    for i, stage in enumerate(trans_config):
        if stage['type'] in tensor_types:
            return i

    return None


def _handle_size_logic(trans_config: List[Dict[str, Any]], size: Optional[int]) -> List[Dict[str, Any]]:
    """
    Handle size logic based on the last key stage:
    1. if 'last key stage' is center_crop, replace its size to [size, size]
    2. if 'last key stage' is resize, replace its size to size
    3. if 'last key stage' is not those, add a resize stage with size after it
    4. if no 'last key stage' found, add one at its place
    """
    if size is None:
        return trans_config

    resize_config = {
        "antialias": True,
        "interpolation": "bicubic",
        "max_size": None,
        "size": size,
        "type": "resize"
    }
    if not trans_config:
        # Case 4: no stages found, add resize
        return [resize_config]

    to_tensor_index = _find_to_tensor_index(trans_config)
    if to_tensor_index is None:
        # No valid last key stage found, insert at the end
        return [*trans_config, resize_config]

    if to_tensor_index > 0:
        last_key_stage = trans_config[to_tensor_index - 1]
        if last_key_stage['type'] == 'resize':
            # Case 2: replace resize size
            last_key_stage['size'] = size
        else:
            # Case 3: add resize stage after last key stage
            trans_config.insert(to_tensor_index, resize_config)
    else:
        trans_config.insert(0, resize_config)

    return trans_config


def _apply_common_transformations(
        trans_config: List[Dict[str, Any]],
        use_pre_align: Optional[bool],
        pre_align_size: int,
        size: Optional[int]
) -> List[Dict[str, Any]]:
    """Apply common transformation logic for both timm and transformers."""
    # Handle pre-align logic
    trans_config = _handle_pre_align_logic(trans_config, use_pre_align, pre_align_size)

    # Handle size operations
    trans_config = _handle_useless_size_operation(trans_config)

    # Handle size logic
    trans_config = _handle_size_logic(trans_config, size)

    # Handle size operations
    trans_config = _handle_useless_size_operation(trans_config)

    return trans_config


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

    trans_config = _apply_common_transformations(trans_config, use_pre_align, pre_align_size, size)
    return ImgutilsBasedImageProcessor(stages=trans_config)


def load_preprocessor_from_transformers(
        model_name: str, use_pre_align: Optional[bool] = None, pre_align_size: int = 512,
        size: Optional[int] = None, **kwargs
):
    logging.info(f'Loading preprocessor of model {model_name!r} with transformers format ...')
    trans = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    if type(trans).__name__ == ImgutilsBasedImageProcessor.__name__:
        trans: ImgutilsBasedImageProcessor
        trans_config = parse_torchvision_transforms(trans.stages)
    else:
        trans_config = parse_torchvision_transforms(create_transforms_from_transformers(trans))

    trans_config = _apply_common_transformations(trans_config, use_pre_align, pre_align_size, size)
    return ImgutilsBasedImageProcessor(stages=trans_config)


def load_preprocessor(model_name: str, hf_token: Optional[str] = None,
                      use_pre_align: Optional[bool] = None, pre_align_size: int = 512,
                      size: Optional[int] = None, **kwargs):
    if os.path.exists(os.path.join(model_name, 'preprocessor_config.json')):
        return load_preprocessor_from_transformers(
            model_name=model_name,
            use_pre_align=use_pre_align,
            pre_align_size=pre_align_size,
            size=size,
            **kwargs
        )
    elif model_name.startswith('hf-hub:'):
        return load_preprocessor_from_timm(
            model_name=model_name,
            use_pre_align=use_pre_align,
            pre_align_size=pre_align_size,
            size=size,
        )
    else:
        if '/' not in model_name:
            return load_preprocessor_from_timm(
                model_name=model_name,
                use_pre_align=use_pre_align,
                pre_align_size=pre_align_size,
                size=size,
            )

        try:
            hf_hub_download(
                repo_id=model_name,
                repo_type='model',
                filename='preprocessor_config.json',
                token=hf_token,
            )
        except (RepositoryNotFoundError, EntryNotFoundError):
            return load_preprocessor_from_timm(
                model_name=f'hf-hub:{model_name}',
                use_pre_align=use_pre_align,
                pre_align_size=pre_align_size,
                size=size,
            )
        else:
            return load_preprocessor_from_transformers(
                model_name=model_name,
                use_pre_align=use_pre_align,
                pre_align_size=pre_align_size,
                size=size,
                **kwargs
            )
