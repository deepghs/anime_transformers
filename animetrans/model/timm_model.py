import copy
from typing import Optional

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
            num_classes: int = 1,
            model_args: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.model_args = dict(model_args or {})


TimmModelConfig.register_for_auto_class()


class TimmModel(PreTrainedModel):
    config_class = TimmModelConfig
    _auto_class = 'AutoModel'

    def __init__(self, config: TimmModelConfig, pretrained: bool = False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        model_name = config.model_name
        model_args = config.model_args
        num_classes = config.num_classes

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
        if model.num_classes != num_classes:
            model.reset_classifier(num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)


TimmModel.register_for_auto_class()
