import copy
from typing import Optional, List

import torch
from torch import nn

from .ensure_timm import ensure_timm_dependency

ensure_timm_dependency()

exec('from timm import create_model as _timm_create_model')
from transformers import PretrainedConfig, PreTrainedModel


class TimmModelConfig(PretrainedConfig):
    model_type = "timm_model"
    _auto_class = "AutoConfig"

    def __init__(
            self,
            model_name: str = 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k',
            task_type: str = 'classification',
            task_config: Optional[dict] = None,
            model_args: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.task_type = task_type
        self.task_config = dict(task_config or {})
        self.model_args = dict(model_args or {})

    @property
    def num_classes(self) -> int:
        if self.task_type == 'classification':
            return len(self.task_config['classes'])
        else:
            raise AttributeError(f'No num classes for {self.task_type} task.')

    @property
    def num_tags(self) -> int:
        if self.task_type == 'tagging':
            return len(self.task_config['tags'])
        else:
            raise AttributeError(f'No num tags for {self.task_type} task.')

    @property
    def num_values(self) -> int:
        if self.task_type == 'regression':
            return len(self.task_config['values'])
        else:
            raise AttributeError(f'No num values for {self.task_type} task.')

    @property
    def num_outputs(self) -> int:
        if self.task_type == 'classification':
            return len(self.task_config['classes'])
        elif self.task_type == 'tagging':
            return len(self.task_config['tags'])
        elif self.task_type == 'regression':
            return len(self.task_config['values'])
        else:
            raise ValueError(f'Unknown task type - {self.task_type!r}.')

    def create_infer_layer(self):
        if self.task_type == 'classification':
            return ClassificationInferHead()
        elif self.task_type == 'tagging':
            return TaggingInferHead()
        elif self.task_type == 'regression':
            mean, std = [], []
            for item in self.task_config['values']:
                mean.append(item['mean'])
                std.append(item['std'])
            return RegressionInferHead(mean, std)
        else:
            raise ValueError(f'Unknown task type - {self.task_type!r}.')


TimmModelConfig.register_for_auto_class()


class ClassificationInferHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)


class TaggingInferHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class RegressionInferHead(nn.Module):
    """
    Denormalization module that restores normalized data to original value range

    Args:
        mean (List[float]): Mean values used during normalization, length n
        std (List[float]): Standard deviation values used during normalization, length n
    """

    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: perform denormalization operation

        Args:
            x (torch.Tensor): Input tensor, last dimension must be n

        Returns:
            torch.Tensor: Denormalized tensor with same shape as input
        """
        # Check the last dimension of input tensor

        # Denormalization formula: original = normalized * std + mean
        # Assumes original normalization formula was: normalized = (original - mean) / std
        return x * self.std + self.mean

    def extra_repr(self) -> str:
        """Return extra information for module printing"""
        return f'n={self.mean.shape[-1]}'


class TimmModel(PreTrainedModel):
    config_class = TimmModelConfig
    _auto_class = 'AutoModel'

    def __init__(self, config: TimmModelConfig, pretrained: bool = False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        model_name = config.model_name
        model_args = config.model_args
        num_outputs = config.num_outputs

        try:
            # noinspection PyUnresolvedReferences
            model = _timm_create_model(model_name=model_name, pretrained=pretrained, **model_args)
        except TypeError:
            if 'img_size' in model_args:  # for some model dont support img_size (like mobilenet)
                _model_args = copy.deepcopy(model_args)
                del _model_args['img_size']
                # noinspection PyUnresolvedReferences
                model = _timm_create_model(model_name=model_name, pretrained=pretrained, **_model_args)
            else:
                raise
        if model.num_classes != num_outputs:
            model.reset_classifier(num_outputs)

        self.model = model

    def forward(self, x):
        return self.model(x)


TimmModel.register_for_auto_class()
