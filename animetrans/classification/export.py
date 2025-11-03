import glob
import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Optional, Literal, List

from PIL import Image
from ditk import logging
from hbutils.encoding import sha3
from hfutils.operate import get_hf_client
from transformers import AutoModel, AutoImageProcessor

from ..dataset import load_pretrained_tag
from ..onnx import export_model_to_onnx, ExportedONNXNotUniqueError
from ..utils import torch_model_profile_via_calflops, is_tensorboard_has_content

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export(workdir: str, repo_id: Optional[str] = None,
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

        class_key = meta_info['class_key']
        image_key = meta_info['image_key']
        task_type = meta_info['task_type']
        if task_type != 'classification':
            raise RuntimeError(f'Workdir {workdir!r} is not a classification task, but {task_type!r} instead.')

        dataset_repo_id = meta_info['train']['dataset']
        checkpoints = os.path.join(workdir, 'checkpoints')
        best_ckpt_dir = os.path.join(checkpoints, 'best')
        logging.info(f'Loading model from {best_ckpt_dir!r} ...')
        model = AutoModel.from_pretrained(best_ckpt_dir, trust_remote_code=True, use_infer_head=True)
        logging.info(f'Model loaded:\n{model!r}')

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

        if not no_onnx_export:
            onnx_file = os.path.join(upload_dir, 'model.onnx')
            logging.info(f'Dumping to onnx file {onnx_file!r} ...')
            try:
                export_model_to_onnx(
                    model=model,
                    dummy_input=dummy_input_test,
                    onnx_filename=onnx_file,
                    metadata={**meta, 'classes': meta_info['classes']},
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
