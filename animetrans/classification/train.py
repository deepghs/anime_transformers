import json
import os.path
import random
from typing import Optional, Tuple

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ditk import logging
from hbutils.random import global_seed

from .dataset import load_classes
from ..dataset import load_pretrained_tag
from ..model import ModelStep


def train(
        workdir: str,
        dataset_repo_id: str,
        model_name: str,
        num_workers: int = 16,
        max_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-3,
        key_metric: str = 'macro_f1',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
        model_args: Optional[dict] = None,

        class_key: str = 'class',
        image_key: str = 'webp',
        use_pretrained_weight: bool = True,
        adam_betas: Optional[Tuple[float, float]] = None,
        use_normalize: bool = False,
):
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    if seed is None:
        seed = random.randint(0, (1 << 31) - 1)
    blist = [seed]
    broadcast_object_list(blist, from_process=0)
    seed = blist[0] + accelerator.process_index
    # native random, numpy, torch and faker's seeds are includes
    # if you need to register more library for seeding, see:
    # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
    logging.info(f'Globally set the random seed {seed!r} in process #{accelerator.process_index}.')
    global_seed(seed)

    os.makedirs(workdir, exist_ok=True)

    classes_info = load_classes(repo_id=dataset_repo_id)
    with open(os.path.join(workdir, 'classes.json'), 'w') as f:
        json.dump(classes_info.classes, f)
    checkpoints = os.path.join(workdir, 'checkpoints')
    last_ckpt_dir = os.path.join(checkpoints, 'last')
    model_args = dict(model_args or {})
    model_args = {**model_args}
    if os.path.exists(os.path.join(last_ckpt_dir, 'model.safetensors')):
        if accelerator.is_main_process:
            logging.info(f'Loading last checkpoint from {last_ckpt_dir!r} ...')
        model_step = ModelStep.load_from_dir(last_ckpt_dir)

        previous_epoch = model_step.epoch
        if accelerator.is_main_process:
            logging.info(f'Resume from epoch {previous_epoch!r}.')
    else:
        if accelerator.is_main_process:
            logging.info(f'No last checkpoint found, initialize {timm_model_name!r} model '
                         f'with {plural_word(len(tags_info.tags), "tag")}.')

        model = Model.new(
            model_name=timm_model_name,
            tags=tags_info.tags,
            pretrained=True,
            pretrained_cfg=pretrained_cfg,
            model_args=model_args,
        )
        previous_epoch = 0

    model: Model
    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
