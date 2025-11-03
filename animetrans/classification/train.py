import json
import os.path
import random
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from ditk import logging
from hbutils.random import global_seed
from hbutils.string import plural_word
from torch.optim import lr_scheduler
from transformers import AutoImageProcessor

from .dataset import load_classes, load_dataloader
from .loss import FocalLoss
from ..dataset import load_pretrained_tag
from ..model import ModelStep, load_model
from ..preprocess import load_preprocessor
from ..session import TrainSession

_DEFAULT_BETAS = (0.9, 0.999)


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
        aug_args: Optional[dict] = None,
        preprocess_args: Optional[dict] = None,
        class_key: str = 'class',
        image_key: str = 'webp',
        adam_betas: Optional[Tuple[float, float]] = None,
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
        model = model_step.model
        previous_epoch = model_step.epoch
        if accelerator.is_main_process:
            logging.info(f'Resume from epoch {previous_epoch!r}.')
    else:
        if accelerator.is_main_process:
            logging.info(f'No last checkpoint found, initialize {model_name!r} model '
                         f'with {plural_word(len(classes_info.classes), "class")}.')
        model = load_model(
            model_name=model_name,
            task_type='classification',
            task_config={'classes': classes_info.classes},
            **model_args,
        )
        previous_epoch = 0

    pretrained_tag = load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
    previous_epoch: int
    preprocess_args = dict(preprocess_args or {})
    aug_args = dict(aug_args or {})
    adam_betas = adam_betas or _DEFAULT_BETAS
    train_cfg = {
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'seed': seed,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'key_metric': key_metric,
        'processes': accelerator.num_processes,
        'dataset': dataset_repo_id,
        'model_args': model_args,
        'aug_args': aug_args,
        'preprocess_args': preprocess_args,
        'pretrained_tag': pretrained_tag,
        'image_key': image_key,
        'class_key': class_key,
        'adam_betas': adam_betas,
        'eval_epoch': eval_epoch,
    }
    if accelerator.is_main_process:
        logging.info(f'Training configurations: {train_cfg!r}.')
        with open(os.path.join(workdir, 'meta.json'), 'w') as f:
            json.dump({
                'model_name': model.model_name,
                'classes': classes_info.classes,
                'model_args': model.model_args,
                'pretrained_cfg': model.pretrained_cfg,
                'train': train_cfg,
            }, f, indent=4, ensure_ascii=False, sort_keys=True)

    preprocessor_dir = os.path.join(workdir, 'preprocessor')
    if os.path.exists(os.path.join(preprocessor_dir, 'preprocessor_config.json')):
        if accelerator.is_main_process:
            logging.info(f'Loading from existing preprocessor {preprocessor_dir!r} ...')
        preprocessor = AutoImageProcessor.from_pretrained(preprocessor_dir, trust_remote_code=True)
    else:
        if accelerator.is_main_process:
            logging.info(f'Creating new preprocessor with model {model_name!r}, args: {preprocess_args!r} ...')
        preprocessor = load_preprocessor(
            model_name=model_name,
            **preprocess_args,
        )
        if accelerator.is_main_process:
            logging.info(f'Saving it to {preprocessor_dir!r} ...')
        preprocessor.save_pretrained(preprocessor_dir)

    train_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        preprocessor=preprocessor,
        class_key=class_key,
        split='train',
        aug_args=aug_args,
        batch_size=batch_size,
        num_workers=num_workers,
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
    )
    eval_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        preprocessor=preprocessor,
        class_key=class_key,
        split='validation',
        aug_args={},
        batch_size=batch_size,
        num_workers=num_workers,
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
    )

    loss_fn = FocalLoss(reduction='none', num_classes=len(classes_info.classes))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model, optimizer, train_dataloader, eval_dataloader, loss_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, loss_fn)

    # scheduler do not need to get prepared
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=max_epochs,
        pct_start=0.15,
        final_div_factor=20.,
    )
    # start from previous LR
    for _ in range(previous_epoch * len(train_dataloader)):
        scheduler.step()

    if accelerator.is_main_process:
        logging.info(f'Model Class: {type(model)!r}')
        session = TrainSession(
            workdir, key_metric=key_metric,
            hyperparams=train_cfg,
            project=f'{dataset_repo_id}',
        )
        logging.info('Training start!')
