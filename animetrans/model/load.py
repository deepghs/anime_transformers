import json
from typing import Optional

from huggingface_hub import hf_hub_download
from timm.models import parse_model_name
from transformers import AutoModel

from .timm_model import TimmModelConfig, TimmModel


def _get_timm_repo_id(timm_model_name: str):
    model_source, model_name = parse_model_name(timm_model_name)
    if model_source == 'hf-hub':
        return model_name
    else:
        return f'timm/{model_name}'


def load_model_from_timm(
        model_name: str, task_type: str = 'classification', task_config: Optional[dict] = None,
        unbind_from_src_repo: bool = True, use_infer_head: bool = False, **model_args,
):
    model_args = dict(model_args or {})
    task_config = dict(task_config or {})
    origin_model_cfg = TimmModelConfig(
        model_name=model_name,
        model_args=model_args,
        task_type=task_type,
        task_config=task_config,
        use_infer_head=use_infer_head,
    )
    origin_model = TimmModel(origin_model_cfg, pretrained=True)

    if unbind_from_src_repo:
        repo_id = _get_timm_repo_id(model_name)
        with open(hf_hub_download(
                repo_id=repo_id,
                repo_type='model',
                filename='config.json'
        ), 'r') as f:
            config = json.load(f)

        model_cfg = TimmModelConfig(
            model_name=config['architecture'],
            model_args={**model_args, **(config.get('model_args') or {})},
            task_type=task_type,
            task_config=task_config,
            use_infer_head=use_infer_head,
        )
        model = TimmModel(model_cfg, pretrained=False)
        model.load_state_dict(origin_model.state_dict())
    else:
        model = origin_model

    return model


def load_model_from_timm_transformers(
        model_name: str, task_type: Optional[str] = None, task_config: Optional[dict] = None,
        use_infer_head: Optional[bool] = None,
):
    origin_model = AutoModel.from_pretrained(model_name, pretrained=True)
    assert type(origin_model).__name__ == 'TimmModel', \
        f'Unsupported source transformers model - {model_name!r}.'
    config = TimmModelConfig(
        model_name=origin_model.config.model_name,
        model_args=origin_model.config.model_args,
        task_type=task_type or origin_model.config.task_type,
        task_config=task_config or origin_model.config.task_config,
        use_infer_head=use_infer_head if use_infer_head is not None else origin_model.config.use_infer_head,
    )
    if origin_model.config.num_outputs != config.num_outputs:
        origin_model.model.reset_classifier(config.num_outputs)

    model = TimmModel(config, pretrained=False)
    model.load_state_dict(origin_model.state_dict())
    return model


def load_model(model_name: str, task_type: Optional[str] = None, task_config: Optional[dict] = None,
               use_infer_head: bool = False, **kwargs):
    if '/' not in model_name or 'hf-hub:' in model_name:
        return load_model_from_timm(
            model_name=model_name,
            task_type=task_type,
            task_config=task_config,
            use_infer_head=use_infer_head,
            **kwargs
        )
    else:
        return load_model_from_timm_transformers(
            model_name=model_name,
            task_type=task_type,
            task_config=task_config,
            use_infer_head=use_infer_head,
        )
