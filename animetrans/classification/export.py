import glob
import json
import os
import re
import shutil
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional, Literal, List

import click
from PIL import Image
from accelerate import Accelerator
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client, upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from thop import clever_format
from timm.models import parse_model_name
from transformers import AutoModel, AutoImageProcessor

from .test import test
from ..dataset import load_pretrained_tag
from ..model import StepInfo
from ..onnx import export_model_to_onnx, ExportedONNXNotUniqueError
from ..utils import torch_model_profile_via_calflops, is_tensorboard_has_content, GLOBAL_CONTEXT_SETTINGS, \
    print_version, VALID_LICENCES

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def _get_timm_repo_id(timm_model_name: str):
    model_source, model_name = parse_model_name(timm_model_name)
    if model_source == 'hf-hub':
        return model_name
    else:
        return f'timm/{model_name}'


def _get_base_model_repo_id(model_name: str):
    if '/' not in model_name or 'hf-hub:' in model_name:
        return _get_timm_repo_id(model_name)
    else:
        return model_name


def export(workdir: str, repo_id: Optional[str] = None, ckpt_name: str = 'best',
           visibility: Literal['private', 'public', 'gated', 'manual'] = 'private',
           logfile_anonymous: bool = True, append_tags: Optional[List[str]] = None,
           title: Optional[str] = None, description: Optional[str] = None, license: str = 'mit',
           onnx_opset_version: int = 14, no_onnx_export: bool = False, namespace: str = 'deepghs'):
    append_tags = list(append_tags or [])
    hf_client = get_hf_client()
    with TemporaryDirectory() as upload_dir:
        meta_info_file = os.path.join(workdir, 'meta.json')
        logging.info(f'Loading meta from {meta_info_file!r} ...')
        with open(meta_info_file, 'r') as f:
            meta_info = json.load(f)

        task_type = meta_info.get('task_type', 'classification')
        if task_type != 'classification':
            raise RuntimeError(f'Workdir {workdir!r} is not a classification task, but {task_type!r} instead.')

        dataset_repo_id = meta_info['train']['dataset']
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_dir = os.path.join(checkpoints, ckpt_name)
        logging.info(f'Loading model from {best_ckpt_dir!r} ...')
        model = AutoModel.from_pretrained(best_ckpt_dir, trust_remote_code=True, use_infer_head=True)
        # logging.info(f'Model loaded:\n{model!r}')
        logging.info(f'Model loaded.')

        preprocessor_dir = os.path.join(workdir, 'preprocessor')
        preprocessor = AutoImageProcessor.from_pretrained(preprocessor_dir, trust_remote_code=True)
        logging.info(f'Preprocessor loaded from {preprocessor_dir!r}:\n{preprocessor!r}')

        pretrained_tag = meta_info['train'].get('pretrained_tag') or load_pretrained_tag(dataset_repo_id)
        logging.info(f'Pretrained tag {pretrained_tag!r} found for dataset {dataset_repo_id!r}.')

        model_name = '.'.join([f'cls-{pretrained_tag}', model.config.model_name, *append_tags])
        repo_id = repo_id or f'{namespace}/{model_name}'
        logging.info(f'Target repository: {repo_id!r}.')
        if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
            hf_client.create_repo(repo_id=repo_id, repo_type='model', private=visibility == 'private')
            if visibility == 'gated':
                hf_client.update_repo_settings(repo_id=repo_id, repo_type='model', gated='auto')
            elif visibility == 'manual':
                hf_client.update_repo_settings(repo_id=repo_id, repo_type='model', gated='manual')

        logging.info(f'Saving model to {upload_dir!r} ...')
        model.save_pretrained(upload_dir)
        logging.info(f'Saving preprocessor to {upload_dir!r} ...')
        preprocessor.save_pretrained(upload_dir)

        image = Image.new('RGB', (1024, 1024), 'white')
        dummy_input_test = preprocessor(image)['pixel_values']
        logging.info(f'Dummy input for model: {dummy_input_test.shape!r}')

        flops, params, macs = torch_model_profile_via_calflops(model=model, input_=dummy_input_test)
        meta_info['flops'] = flops
        meta_info['params'] = params
        meta_info['macs'] = macs
        new_meta_file = os.path.join(upload_dir, 'meta.json')
        logging.info(f'Saving metadata to {new_meta_file!r} ...')
        with open(new_meta_file, 'w') as f:
            json.dump(meta_info, f, indent=4, sort_keys=True, ensure_ascii=False)

        eval_step_info = StepInfo.load_from_dir(best_ckpt_dir)
        test_dir = os.path.join(workdir, 'checkpoints', ckpt_name, 'test')
        if os.path.join(os.path.join(test_dir, 'metrics.json')):
            test_step_info = StepInfo.load_from_dir(test_dir)
        else:
            test_step_info = None

        classes = meta_info['classes']
        mark_info = {
            'epoch': eval_step_info.epoch,
            'task_type': 'classification',
            'classes': classes,
            **{
                f'eval/{key}': value
                for key, value in eval_step_info.metrics.items()
                if not isinstance(value, Image.Image)
            },
            **{
                f'test/{key}': value
                for key, value in (test_step_info or {}).metrics.items()
                if not isinstance(value, Image.Image)
            },
            **{
                f'train/{key}': value
                for key, value in meta_info['train'].items()
            },
            'copyright': 'DeepGHS (https://github.com/deepghs)',
            'project': 'Anime Transformers',
            'source': 'https://github.com/deepghs/anime_transformers',
        }

        if not no_onnx_export:
            onnx_file = os.path.join(upload_dir, 'model.onnx')
            logging.info(f'Dumping to onnx file {onnx_file!r} ...')
            try:
                export_model_to_onnx(
                    model=model,
                    dummy_input=dummy_input_test,
                    onnx_filename=onnx_file,
                    metadata=mark_info,
                    opset_version=onnx_opset_version,
                    verbose=False,
                )
            except ExportedONNXNotUniqueError:
                logging.exception('Non-unique exported ONNX files, so onnx uploading will be disabled.')
                no_onnx_export = True

        for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
            if not is_tensorboard_has_content(logfile):
                logging.warning(f'Tensorboard file {logfile!r} is empty, skipped.')
                continue

            logging.info(f'Tensorboard file {logfile!r} found.')
            matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
            assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

            timestamp = matching.group('timestamp')
            machine = matching.group('machine')
            if logfile_anonymous:
                machine = sha3(machine.encode(), n=224)
            extra = matching.group('extra')

            final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
            dst_log_file = os.path.join(upload_dir, final_name)
            logging.info(f'Adding log file {logfile!r} to {dst_log_file!r} ...')
            shutil.copyfile(logfile, dst_log_file)

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            base_model_repo_id = meta_info['train'].get('model_name')
            if base_model_repo_id:
                if base_model_repo_id == repo_id:
                    base_models = hf_client.repo_info(repo_id=repo_id, repo_type='model').card_data.get(
                        'base_model') or []
                    if base_models:
                        base_model_repo_id = base_models[0]

            print(f'---', file=f)
            print(f'tags:', file=f)
            print(f'- image-classification', file=f)
            print(f'- transformers', file=f)
            print(f'- animetrans', file=f)
            print(f'- dghs-imgutils', file=f)
            print(f'library_name: timm', file=f)
            print(f'license: {license}', file=f)
            print(f'datasets:', file=f)
            print(f'- {dataset_repo_id}', file=f)
            if base_model_repo_id:
                print(f'base_model:', file=f)
                print(f'- {base_model_repo_id}', file=f)
            print(f'---', file=f)
            print(f'', file=f)

            title = title or f'Anime Classifier {repo_id}'
            print(f'# {title}', file=f)
            print(f'', file=f)
            if description:
                print(f'{description}', file=f)
                print(f'', file=f)

            s_flops, s_params, s_macs = clever_format([flops, params, macs], "%.1f")
            print(f'## Model Details', file=f)
            print(f'', file=f)
            print(f'- **Model Type:** Image Classification', file=f)
            print(f'- **Model Stats:**', file=f)
            print(f'  - Params: {s_params}', file=f)
            print(f'  - FLOPs / MACs: {s_flops} / {s_macs}', file=f)
            print(f'  - Image size: train = {dummy_input_test.shape[-1]} x {dummy_input_test.shape[-2]}', file=f)
            print(f'- **Dataset:** [{dataset_repo_id}]'
                  f'({hf_hub_repo_url(repo_id=dataset_repo_id, repo_type="dataset", endpoint="https://huggingface.co")})',
                  file=f)

            print(f'  - Classes: {", ".join(map(lambda x: f"`{x}`", classes))}', file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Upload model {repo_id!r}',
            clear=True,
        )


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Calculating test metrics for multilabel taggers.")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetrans.classification.export'), expose_value=False, is_eager=True)
@click.option('--num-workers', '-nw', default=32, type=int, help='Number of workers', show_default=True)
@click.option('--batch-size', '-bs', default=32, type=int, help='Batch size', show_default=True)
@click.option('--workdir', '-w', default=None, type=str, help='Workdir to save training data', show_default=True)
@click.option('--non-force/--force', default=False, help='Force re-calculate.', show_default=True)
@click.option('--need-metrics/--no-metrics', default=True, help='Need metrics to get tested.', show_default=True)
@click.option('--visibility', '-V', default='manual', type=click.Choice(['private', 'public', 'gated', 'manual']),
              help='Visibility when creating model repository (will be ignored when model repository already exist.',
              show_default=True)
@click.option('--repository', '-r', default=None, help='Repository for uploading model', show_default=True)
@click.option('--tag', '-t', 'tags', multiple=True, type=str, help='Append tags for repository name', show_default=True)
@click.option('--title', '-T', default=None, type=str, help='Title for repository', show_default=True)
@click.option('--description', '-desc', default=None, type=str, help='Description for repository', show_default=True)
@click.option('-l', '--licence', '--license', 'license', type=click.Choice(VALID_LICENCES), default='mit',
              help='Licence for repository.', show_default=True)
@click.option('-opv', '--onnx-opset-version', 'onnx_opset_version', default=14, type=int,
              help='OpSet Version of ONNX Export.', show_default=True)
@click.option('--no-onnx-export', 'no_onnx_export', is_flag=True, default=False, type=bool,
              help='No ONNX model to export, just save the weights.', show_default=True)
@click.option('-ns', '--namespace', 'namespace', default='deepghs', type=str, show_default=True,
              help='Namespace for the publish repository')
@click.option('--ckpt-name', '-c', 'ckpt_name', default='best', help='Name of the checkpoint to test',
              show_default=True)
def cli(workdir, num_workers, batch_size, force, need_metrics, repository, visibility, tags, title, description,
        license, ckpt_name, onnx_opset_version, no_onnx_export, namespace):
    logging.try_init_root(logging.INFO)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    if need_metrics:
        test(
            workdir=workdir,
            num_workers=num_workers,
            batch_size=batch_size,
            force=force,
            ckpt_name=ckpt_name,
            accelerator=accelerator,
        )

    if accelerator.is_main_process:
        export(
            workdir=workdir,
            repo_id=repository,
            visibility=visibility,
            logfile_anonymous=True,
            append_tags=tags,
            title=title,
            description=description,
            license=license,
            onnx_opset_version=onnx_opset_version,
            no_onnx_export=no_onnx_export,
            namespace=namespace,
            ckpt_name=ckpt_name,
        )


if __name__ == '__main__':
    cli()
