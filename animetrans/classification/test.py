import json
import os
from functools import partial
from pprint import pformat
from typing import Optional

import click
import torch
from accelerate import Accelerator
from ditk import logging
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, top_k_accuracy_score
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

from .dataset import load_dataloader, load_classes
from .plot_cm import plt_confusion_matrix
from .plot_export import plt_export
from .plot_pr import plt_multiclass_metrics, plt_f1_scores
from ..dataset import load_pretrained_tag
from ..model import ModelStep, StepInfo
from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version


def test(workdir: str, num_workers: int = 32, batch_size: int = 32, force: bool = False,
         accelerator: Optional[Accelerator] = None, ckpt_name: str = 'best'):
    model_ckpt_dir = os.path.join(workdir, 'checkpoints', ckpt_name)
    eval_step_info = StepInfo.load_from_dir(model_ckpt_dir)
    metrics_dir = os.path.join(model_ckpt_dir, 'test')

    if not force and os.path.exists(os.path.join(metrics_dir, 'metrics.json')):
        test_step_info = StepInfo.load_from_dir(metrics_dir)
        if test_step_info.epoch == eval_step_info.epoch:
            logging.info(f'Already checkpoint {ckpt_name!r} tested for {workdir}, skipped.')
            return

    accelerator = accelerator or Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    image_key, class_key = meta_info['train']['image_key'], meta_info['train']['class_key']
    dataset_repo_id = meta_info['train']['dataset']
    classes_info = load_classes(repo_id=dataset_repo_id)
    pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
    logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')
    num_topk = meta_info['train'].get('num_topk')
    if num_topk is None:
        num_topk = max(min(len(classes_info.classes) // 2, 5), 1)
    key_metric = meta_info['train']['key_metric']
    accelerator.wait_for_everyone()

    model_step = ModelStep.load_from_dir(model_ckpt_dir)
    epoch = model_step.epoch
    if accelerator.is_main_process:
        logging.info(f'Model loaded from {model_ckpt_dir!r}, '
                     f'with key metric {key_metric!r} value {model_step.metrics[key_metric]!r}.')

    model = AutoModel.from_pretrained(model_ckpt_dir, trust_remote_code=True)
    if accelerator.is_main_process:
        # logging.info(f'Model loaded:\n{model!r}')
        logging.info(f'Model loaded.')

    preprocessor_dir = os.path.join(workdir, 'preprocessor')
    preprocessor = AutoImageProcessor.from_pretrained(preprocessor_dir, trust_remote_code=True)
    if accelerator.is_main_process:
        logging.info(f'Preprocessor loaded from {preprocessor_dir!r}:\n{preprocessor!r}')

    test_dataloader = load_dataloader(
        repo_id=dataset_repo_id,
        preprocessor=preprocessor,
        class_key=class_key,
        split='test',
        aug_args={},
        batch_size=batch_size,
        num_workers=num_workers,
        is_main_process=accelerator.is_main_process,
        image_key=image_key,
    )

    infer_head = model.config.create_infer_head()
    model, infer_head, test_dataloader = accelerator.prepare(model, infer_head, test_dataloader)
    if accelerator.is_main_process:
        logging.info(f'Model Class: {type(model)!r}')
        logging.info('Testing start!')

    infer_head.eval()
    model.eval()

    with torch.no_grad():
        model.eval()

        with torch.no_grad():
            test_total = 0
            y_true, y_pred, y_score = [], [], []

            for i, (inputs, labels_) in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
                inputs = inputs.float()
                labels_ = labels_

                outputs = model(inputs)
                test_total += labels_.shape[0]

                y_true.append(labels_.clone().detach())
                y_pred.append(torch.argmax(outputs, dim=-1).detach())
                y_score.append(infer_head(outputs).detach())

                if i % 10 == 0:
                    accelerator.wait_for_everyone()

            accelerator.wait_for_everyone()

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            y_score = torch.concat(y_score)

            y_true = accelerator.gather(y_true).detach().cpu().numpy()
            y_pred = accelerator.gather(y_pred).detach().cpu().numpy()
            y_score = accelerator.gather(y_score).detach().cpu().numpy()

            if accelerator.is_main_process:
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
                macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
                macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
                if len(classes_info.classes) > 2:
                    macro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                else:
                    macro_auc = roc_auc_score(y_true, y_score[:, 1], average='macro')

                micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0.0)
                micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0.0)
                micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0.0)
                if len(classes_info.classes) > 2:
                    micro_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')
                else:
                    micro_auc = roc_auc_score(y_true, y_score[:, 1], average='micro')

                metrics = {
                    'accuracy': top_k_accuracy_score(
                        y_true,
                        y_score if len(classes_info.classes) > 2 else y_score[:, 1],
                        labels=list(range(len(classes_info.classes))), k=1
                    ),
                    f'top-{num_topk}': top_k_accuracy_score(
                        y_true,
                        y_score if len(classes_info.classes) > 2 else y_score[:, 1],
                        labels=list(range(len(classes_info.classes))), k=num_topk
                    ),
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
                        title=f'Confusion Matrix on Test #{epoch}',
                        normalize='true',
                    ),
                    'plt_pr': plt_export(
                        plt_multiclass_metrics,
                        y_true, y_score,
                        labels=classes_info.classes,
                        title=f'Precision-Recall Curves on Test #{epoch}',
                    ),
                    'plt_f1': plt_export(
                        plt_f1_scores,
                        y_true, y_score,
                        labels=classes_info.classes,
                        title=f'F1 Scores vs Threshold on Test #{epoch}'
                    ),
                }
                logging.info(f'Test complete, result:\n{pformat(metrics)}')

                os.makedirs(metrics_dir, exist_ok=True)
                StepInfo(epoch=epoch, metrics=metrics).save_to_dir(metrics_dir)
                logging.info(f'Test metrics saved at {metrics_dir!r}')


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Calculating test metrics for multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetimm.multilabel.test'), expose_value=False, is_eager=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--non-force/--force', 'force', default=False, help='Force re-calculate.', show_default=True)
@click.option('--ckpt-name', '-c', 'ckpt_name', default='best', help='Name of the checkpoint to test',
              show_default=True)
def cli(workdir, num_workers, batch_size, force, ckpt_name):
    logging.try_init_root(logging.INFO)
    test(
        workdir=workdir,
        num_workers=num_workers,
        batch_size=batch_size,
        force=force,
        ckpt_name=ckpt_name,
    )


if __name__ == '__main__':
    cli()
