import os
from typing import Optional, Mapping, Union, Dict

from PIL import Image
from ditk import logging

from .base import BaseLogger
from ..model import ModelStep, TimmModel


class CheckpointLogger(BaseLogger):
    def __init__(self, workdir: str, key_metric: str = 'accuracy', save_step_when_better: bool = False, **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.key_metric = key_metric

        self.ckpt_dir = os.path.join(self.workdir, 'checkpoints')
        self._last_step: Optional[int] = None
        self._best_metric_value: Optional[float] = None
        self._save_step_when_better = save_step_when_better
        self._load_last()
        self._load_best()

    @classmethod
    def _load_model_step(cls, directory: str):
        return ModelStep.load_from_dir(directory)

    def _save_model_step(self, directory: str, model: ModelStep):
        model.save_to_dir(directory=directory)

    @property
    def _last_step_dir(self):
        return os.path.join(self.ckpt_dir, 'last')

    def _step_dir(self, step: int):
        return os.path.join(self.ckpt_dir, f'step-{step:06d}')

    def _load_last(self):
        if os.path.exists(self._last_step_dir):
            model_step = self._load_model_step(self._last_step_dir)
            self._last_step = model_step.epoch
            logging.info(f'Last ckpt found at {self._last_step}, with previous step {self._last_step}')
        else:
            self._last_step = None
            logging.info('No last ckpt found.')

    def _save_last(self, model: ModelStep):
        self._save_model_step(self._last_step_dir, model)
        self._last_step = model.epoch
        logging.info(f'Last ckpt model epoch {model.epoch} saved')

    @property
    def _best_step_dir(self):
        return os.path.join(self.ckpt_dir, 'best')

    def _load_best(self):
        if os.path.exists(self._best_step_dir):
            model_step = self._load_model_step(self._best_step_dir)
            self._best_metric_value = model_step.metrics[self.key_metric]
            step = model_step.epoch
            logging.info(f'Best ckpt found at {self._best_step_dir}, '
                         f'with step {step} and {self.key_metric} {self._best_metric_value:.3f}')
        else:
            self._best_metric_value = None
            logging.info('No best ckpt found.')

    def _save_best(self, model: ModelStep):
        if self._best_metric_value is None or model.metrics[self.key_metric] > self._best_metric_value:
            self._save_model_step(self._best_step_dir, model)
            if self._save_step_when_better:
                self._save_model_step(self._step_dir(model.epoch), model)
            self._best_metric_value = model.metrics[self.key_metric]
            logging.info(f'Best ckpt model epoch {model.epoch} saved, '
                         f'with {self.key_metric}\'s new value {self._best_metric_value:.3f}')

    def train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        pass

    def eval_log(self, global_step, model: TimmModel, metrics: Dict[str, Union[float, Image.Image]]):
        model_step = ModelStep(
            model=model,
            epoch=global_step,
            metrics=metrics,
        )
        self._save_last(model_step)
        self._save_best(model_step)
