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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, top_k_accuracy_score
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import AutoImageProcessor

from .dataset import load_classes, load_dataloader
from .loss import FocalLoss
from .plot_cm import plt_confusion_matrix
from .plot_export import plt_export
from .plot_pr import plt_multiclass_metrics, plt_f1_scores
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
        learning_rate: float = 1e-4,
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
        num_topk: Optional[int] = None,
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
    if num_topk is None:
        num_topk = max(min(len(classes_info.classes) // 2, 5), 1)
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

    infer_head = model.config.create_infer_head()
    infer_head.eval()
    model, infer_head, optimizer, train_dataloader, eval_dataloader, loss_fn = \
        accelerator.prepare(model, infer_head, optimizer, train_dataloader, eval_dataloader, loss_fn)

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

    accelerator.wait_for_everyone()

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        if accelerator.is_local_main_process:
            logging.info(f'Training for epoch {epoch!r}')
        model.train()
        train_lr = scheduler.get_last_lr()[0]
        train_loss, train_total = 0.0, 0
        y_true, y_pred, y_score = [], [], []

        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
            inputs = inputs.float()
            labels_ = labels_

            optimizer.zero_grad()
            outputs = model(inputs)
            train_total += labels_.shape[0]

            with torch.no_grad():
                y_true.append(labels_.clone().detach())
                y_pred.append(torch.argmax(outputs, dim=-1).detach())
                y_score.append(infer_head(outputs).detach())

            loss = loss_fn(outputs, labels_).sum()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        accelerator.wait_for_everyone()

        with torch.no_grad():
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            y_score = torch.concat(y_score)

            train_loss = accelerator.gather(
                torch.tensor([train_loss], device=accelerator.device)).sum().detach().cpu().item()
            train_total = accelerator.gather(
                torch.tensor([train_total], device=accelerator.device)).sum().detach().cpu().item()

            y_true = accelerator.gather(y_true).detach().cpu().numpy()
            y_pred = accelerator.gather(y_pred).detach().cpu().numpy()
            y_score = accelerator.gather(y_score).detach().cpu().numpy()

        if accelerator.is_main_process:
            with torch.no_grad():
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
                macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
                macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
                macro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')

                micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0.0)
                micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0.0)
                micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0.0)
                micro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')

            session.tb_train_log(
                global_step=epoch,
                metrics={
                    'loss': train_loss / train_total,
                    'accuracy': top_k_accuracy_score(y_true, y_score, k=1),
                    f'top-{num_topk}': top_k_accuracy_score(y_true, y_score, k=num_topk),
                    'macro_f1': macro_f1,
                    'macro_precision': macro_precision,
                    'macro_recall': macro_recall,
                    'macro_auc': macro_auc,
                    'micro_f1': micro_f1,
                    'micro_precision': micro_precision,
                    'micro_recall': micro_recall,
                    'micro_auc': micro_auc,
                    'learning_rate': train_lr,
                    'plt_confusion': plt_export(
                        plt_confusion_matrix,
                        y_true, y_pred,
                        labels=classes_info.classes,
                        title=f'Confusion Matrix on Train #{epoch}',
                        normalize='true',
                    ),
                    'plt_pr': plt_export(
                        plt_multiclass_metrics,
                        y_true, y_score,
                        labels=classes_info.classes,
                        title=f'Precision-Recall Curves on Train #{epoch}',
                    ),
                    'plt_f1': plt_export(
                        plt_f1_scores,
                        y_true, y_score,
                        labels=classes_info.classes,
                        title=f'F1 Scores vs Threshold on Train #{epoch}'
                    ),
                }
            )

        if epoch % eval_epoch == 0:
            model.eval()

            with torch.no_grad():
                eval_loss, eval_total = 0.0, 0
                y_true, y_pred, y_score = [], [], []

                for i, (inputs, labels_) in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_main_process)):
                    inputs = inputs.float()
                    labels_ = labels_

                    outputs = model(inputs)
                    eval_total += labels_.shape[0]

                    y_true.append(labels_.clone().detach())
                    y_pred.append(torch.argmax(outputs, dim=-1).detach())
                    y_score.append(infer_head(outputs).detach())

                    loss = loss_fn(outputs, labels_).sum()
                    eval_loss += loss.item()

                    if i % 10 == 0:
                        accelerator.wait_for_everyone()

                accelerator.wait_for_everyone()

                y_true = torch.concat(y_true)
                y_pred = torch.concat(y_pred)
                y_score = torch.concat(y_score)

                eval_loss = accelerator.gather(
                    torch.tensor([eval_loss], device=accelerator.device)).sum().detach().cpu().item()
                eval_total = accelerator.gather(
                    torch.tensor([eval_total], device=accelerator.device)).sum().detach().cpu().item()

                y_true = accelerator.gather(y_true).detach().cpu().numpy()
                y_pred = accelerator.gather(y_pred).detach().cpu().numpy()
                y_score = accelerator.gather(y_score).detach().cpu().numpy()

                if accelerator.is_main_process:
                    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
                    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
                    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
                    macro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')

                    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0.0)
                    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0.0)
                    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0.0)
                    micro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')

                    session.tb_eval_log(
                        global_step=epoch,
                        model=model,
                        metrics={
                            'loss': eval_loss / eval_total,
                            'accuracy': top_k_accuracy_score(y_true, y_score, k=1),
                            f'top-{num_topk}': top_k_accuracy_score(y_true, y_score, k=num_topk),
                            'macro_f1': macro_f1,
                            'macro_precision': macro_precision,
                            'macro_recall': macro_recall,
                            'macro_auc': macro_auc,
                            'micro_f1': micro_f1,
                            'micro_precision': micro_precision,
                            'micro_recall': micro_recall,
                            'micro_auc': micro_auc,
                            'plt_confusion': plt_export(
                                plt_confusion_matrix,
                                y_true, y_pred,
                                labels=classes_info.classes,
                                title=f'Confusion Matrix on Eval #{epoch}',
                                normalize='true',
                            ),
                            'plt_pr': plt_export(
                                plt_multiclass_metrics,
                                y_true, y_score,
                                labels=classes_info.classes,
                                title=f'Precision-Recall Curves on Eval #{epoch}',
                            ),
                            'plt_f1': plt_export(
                                plt_f1_scores,
                                y_true, y_score,
                                labels=classes_info.classes,
                                title=f'F1 Scores vs Threshold on Eval #{epoch}'
                            ),
                        }
                    )


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    train(
        workdir='runs/mb_test_10k',
        dataset_repo_id='deepghs/ai-check-10k',
        model_name='hf-hub:animetimm/mobilenetv4_conv_aa_large.dbv4-full',
    )
