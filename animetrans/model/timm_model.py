"""
This module provides a wrapper for TIMM (Torch Image Models) models with support for different computer vision tasks.

The module integrates TIMM models with the Hugging Face transformers library, enabling easy configuration
and deployment of pre-trained vision models for classification, tagging, and regression tasks. It includes
specialized inference heads for different task types and proper configuration management.
"""
import copy
from typing import Optional, List

import torch
from torch import nn

from .ensure_timm import ensure_timm_dependency

ensure_timm_dependency()

exec('from timm import create_model as _timm_create_model')
from transformers import PretrainedConfig, PreTrainedModel


class TimmModelConfig(PretrainedConfig):
    """
    Configuration class for TIMM model wrapper.

    This class extends PretrainedConfig to provide configuration for TIMM models
    with support for different computer vision tasks including classification,
    tagging, and regression.

    :param model_name: Name of the TIMM model to use.
    :type model_name: str
    :param task_type: Type of task ('classification', 'tagging', or 'regression').
    :type task_type: str
    :param task_config: Configuration specific to the task type.
    :type task_config: Optional[dict]
    :param model_args: Additional arguments to pass to the TIMM model.
    :type model_args: Optional[dict]
    """

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
        """
        Get the number of classes for classification tasks.

        :return: Number of classes.
        :rtype: int
        :raises AttributeError: If task type is not classification.
        """
        if self.task_type == 'classification':
            return len(self.task_config['classes'])
        else:
            raise AttributeError(f'No num classes for {self.task_type} task.')

    @property
    def classes(self) -> List[str]:
        """
        Get the list of class names for classification tasks.

        :return: List of class names.
        :rtype: List[str]
        :raises AttributeError: If task type is not classification.
        """
        if self.task_type == 'classification':
            return self.task_config['classes']
        else:
            raise AttributeError(f'No classes for {self.task_type} task.')

    @property
    def num_tags(self) -> int:
        """
        Get the number of tags for tagging tasks.

        :return: Number of tags.
        :rtype: int
        :raises AttributeError: If task type is not tagging.
        """
        if self.task_type == 'tagging':
            return len(self.task_config['tags'])
        else:
            raise AttributeError(f'No num tags for {self.task_type} task.')

    @property
    def tags(self) -> List[str]:
        """
        Get the list of tag names for tagging tasks.

        :return: List of tag names.
        :rtype: List[str]
        :raises AttributeError: If task type is not tagging.
        """
        if self.task_type == 'tagging':
            return self.task_config['tags']
        else:
            raise AttributeError(f'No tags for {self.task_type} task.')

    @property
    def num_values(self) -> int:
        """
        Get the number of values for regression tasks.

        :return: Number of regression values.
        :rtype: int
        :raises AttributeError: If task type is not regression.
        """
        if self.task_type == 'regression':
            return len(self.task_config['values'])
        else:
            raise AttributeError(f'No num values for {self.task_type} task.')

    @property
    def values(self) -> List[dict]:
        """
        Get the list of value configurations for regression tasks.

        :return: List of value configuration dictionaries.
        :rtype: List[dict]
        :raises AttributeError: If task type is not regression.
        """
        if self.task_type == 'regression':
            return self.task_config['values']
        else:
            raise AttributeError(f'No values for {self.task_type} task.')

    @property
    def value_names(self) -> List[str]:
        """
        Get the list of value names for regression tasks.

        :return: List of value names.
        :rtype: List[str]
        :raises AttributeError: If task type is not regression.
        """
        if self.task_type == 'regression':
            return [item['name'] for item in self.task_config['values']]
        else:
            raise AttributeError(f'No value names for {self.task_type} task.')

    @property
    def num_outputs(self) -> int:
        """
        Get the number of outputs for any task type.

        This is a unified property that returns the appropriate output dimension
        based on the task type.

        :return: Number of outputs.
        :rtype: int
        :raises ValueError: If task type is unknown.
        """
        if self.task_type == 'classification':
            return len(self.task_config['classes'])
        elif self.task_type == 'tagging':
            return len(self.task_config['tags'])
        elif self.task_type == 'regression':
            return len(self.task_config['values'])
        else:
            raise ValueError(f'Unknown task type - {self.task_type!r}.')

    def create_infer_layer(self):
        """
        Create the appropriate inference layer based on task type.

        This method instantiates the correct inference head module for the
        configured task type, handling post-processing of model outputs.

        :return: Inference layer module.
        :rtype: nn.Module
        :raises ValueError: If task type is unknown.
        """
        if self.task_type == 'classification':
            return ClassificationInferHead(self.classes)
        elif self.task_type == 'tagging':
            return TaggingInferHead(self.tags)
        elif self.task_type == 'regression':
            mean, std = [], []
            for item in self.task_config['values']:
                mean.append(item['mean'])
                std.append(item['std'])
            return RegressionInferHead(self.value_names, mean, std)
        else:
            raise ValueError(f'Unknown task type - {self.task_type!r}.')


TimmModelConfig.register_for_auto_class()


class ClassificationInferHead(nn.Module):
    """
    Inference head for classification tasks.

    This module applies softmax activation to convert logits to probabilities
    for multi-class classification tasks.

    :param classes: List of class names for the classification task.
    :type classes: List[str]
    """

    def __init__(self, classes: List[str]):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.classes = classes
        self.n_classes = len(classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax activation to input logits.

        :param x: Input logits tensor.
        :type x: torch.Tensor
        :return: Probability distribution over classes.
        :rtype: torch.Tensor
        """
        return self.softmax(x)

    def extra_repr(self) -> str:
        """
        Return extra information for module printing.

        :return: String representation of module parameters.
        :rtype: str
        """
        return f'n_classes={self.n_classes!r}, classes={self.classes!r}'


class TaggingInferHead(nn.Module):
    """
    Inference head for tagging tasks.

    This module applies sigmoid activation to convert logits to probabilities
    for multi-label tagging tasks where multiple tags can be active simultaneously.

    :param tags: List of tag names for the tagging task.
    :type tags: List[str]
    """

    def __init__(self, tags: List[str]):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tags = tags
        self.n_tags = len(tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sigmoid activation to input logits.

        :param x: Input logits tensor.
        :type x: torch.Tensor
        :return: Independent probabilities for each tag.
        :rtype: torch.Tensor
        """
        return self.sigmoid(x)

    def extra_repr(self) -> str:
        """
        Return extra information for module printing.

        :return: String representation of module parameters.
        :rtype: str
        """
        return f'n_tags={self.n_tags!r}, tags={self.tags!r}'


class RegressionInferHead(nn.Module):
    """
    Denormalization module that restores normalized data to original value range.

    This inference head is used for regression tasks where the target values
    were normalized during training. It applies the inverse transformation
    to convert model outputs back to the original scale.

    :param value_names: List of value names for the regression targets.
    :type value_names: List[str]
    :param mean: Mean values used during normalization, length n.
    :type mean: List[float]
    :param std: Standard deviation values used during normalization, length n.
    :type std: List[float]
    """

    def __init__(self, value_names: List[str], mean: List[float], std: List[float]):
        super().__init__()
        self.value_names = value_names
        self.n = len(value_names)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: perform denormalization operation.

        Applies the inverse normalization transformation using the formula:
        original = normalized * std + mean

        :param x: Input tensor, last dimension must match the number of regression targets.
        :type x: torch.Tensor
        :return: Denormalized tensor with same shape as input.
        :rtype: torch.Tensor
        """
        # Denormalization formula: original = normalized * std + mean
        # Assumes original normalization formula was: normalized = (original - mean) / std
        return x * self.std + self.mean

    def extra_repr(self) -> str:
        """
        Return extra information for module printing.

        :return: String representation of module parameters.
        :rtype: str
        """
        return (f'n={self.n!r}, value_names={self.value_names!r}, '
                f'mean={self.mean!r}, std={self.std!r}')


class TimmModel(PreTrainedModel):
    """
    Wrapper class for TIMM models with Hugging Face integration.

    This class provides a bridge between TIMM models and the Hugging Face
    transformers ecosystem, enabling easy loading, configuration, and deployment
    of pre-trained vision models for various computer vision tasks.

    :param config: Configuration object containing model and task settings.
    :type config: TimmModelConfig
    :param pretrained: Whether to load pretrained weights.
    :type pretrained: bool
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TIMM model.

        :param x: Input tensor (typically images).
        :type x: torch.Tensor
        :return: Model output logits.
        :rtype: torch.Tensor
        """
        return self.model(x)


TimmModel.register_for_auto_class()
