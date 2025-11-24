import datetime
import glob
import json
import math
import mimetypes
import os
import random
import re
import tarfile
import textwrap
from functools import partial
from threading import Lock
from typing import Optional

import click
import pandas as pd
from ditk import logging
from hbutils.encoding import int_hash
from hbutils.random import global_seed
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from hfutils.repository import hf_hub_repo_url
from hfutils.utils import hf_normpath, number_to_tag
from huggingface_hub import HfFileSystem, HfApi
from imgutils.data import load_image
from natsort import natsorted
from tqdm import tqdm

from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version, VALID_LICENCES, parallel_call


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help="Convert dataset formats")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetrans.classification.convert'), expose_value=False, is_eager=True)
def cli():
    logging.try_init_root(logging.INFO)


@cli.command('id2datasets', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help="Convert dataset formats from imagefolder to datasets")
@click.option('-v', '--version', is_flag=True,
              callback=partial(print_version, 'animetrans.classification.convert'), expose_value=False, is_eager=True)
@click.option('-i', '--image-folder', type=str, required=True,
              help='Image folder to convert', show_default=True)
@click.option('--min-image-class', type=int, default=50, help='Minimum image count for each class', show_default=True)
@click.option('--eval-percentile', type=int, default=10, help='Percentile of eval set', show_default=True)
@click.option('--test-percentile', type=int, default=10, help='Percentile of test set', show_default=True)
@click.option('-r', '--repository', type=str, required=True,
              help='Repository to dump the datasets format dataset', show_default=True)
@click.option('-a', '--author', type=str, default='narugo1992', help='Author of this dataset', show_default=True)
@click.option('-t', '--title', type=str, default=None, help='Title of this dataset', show_default=True)
@click.option('-pt', '--pretrained_tag', type=str, default=None, help='Pretrained tag of this dataset',
              show_default=True)
@click.option('-bs', '--batch_size', type=int, default=100000, help='Batch size of each package', show_default=True)
@click.option('-n', '--max-workers', type=int, default=32, help='Workers of the data processing', show_default=True)
@click.option('--min-size', type=int, default=640, help='Value of min(height, width)', show_default=True)
@click.option('-l', '--licence', '--license', 'license', type=click.Choice(VALID_LICENCES), default='mit',
              help='Licence for repository.', show_default=True)
def id2datasets(image_folder: str, min_image_class: int, eval_percentile: int, test_percentile: int,
                repository: str, author: str, title: Optional[str], pretrained_tag: Optional[str], batch_size: int,
                max_workers: int, min_size: int, license: str):
    logging.try_init_root(logging.INFO)

    mimetypes.add_type('image/webp', '.webp')
    try:
        import pillow_avif
    except (ImportError, ModuleNotFoundError) as err:
        logging.warning(f'Pillow Avif not launched - {err!r}.')
    else:
        logging.info('Pillow Avif launched.')

    d_class_splits = {}
    records = []
    for name in tqdm(os.listdir(image_folder), desc='Scanning classes'):
        if os.path.isdir(os.path.join(image_folder, name)):
            d_class_splits[name] = {'train': [], 'val': [], 'test': []}
            for file in tqdm(glob.glob(os.path.join(image_folder, name, '**', '*'), recursive=True),
                             desc=f'Scanning class {name!r}'):
                mimetype, _ = mimetypes.guess_type(file)
                if mimetype and mimetype.startswith('image/'):
                    path = hf_normpath(os.path.relpath(file, image_folder))
                    suffix_id = int_hash(path) % 100
                    if suffix_id >= 100 - test_percentile:
                        split = 'test'
                    elif suffix_id >= 100 - test_percentile - eval_percentile:
                        split = 'val'
                    else:
                        split = 'train'
                    d_class_splits[name][split].append(path)
                    records.append({'id': path, 'split': split, 'class': name})

    d_class_splits = {
        name: value for name, value in d_class_splits.items()
        if len(value['train']) + len(value['val']) + len(value['test']) >= min_image_class
    }
    classes = natsorted(d_class_splits.keys())
    df_classes = pd.DataFrame([
        {
            'class': name,
            'total_count': len(d_class_splits[name]['train']) + len(d_class_splits[name]['val']) + len(
                d_class_splits[name]['test']),
            'train_count': len(d_class_splits[name]['train']),
            'val_count': len(d_class_splits[name]['val']),
            'test_count': len(d_class_splits[name]['test']),

        }
        for name in classes
    ])
    logging.info(f'Classes analysis:\n{df_classes}')

    date = datetime.datetime.now()
    safe_title = re.sub(r'[\W_]+', '', title).strip().lower()
    citation = textwrap.dedent(
        f"""
            @inproceedings{{{pretrained_tag or safe_title},
              author = {{{author}}},
              title = {{{title}}},
              year = {{{date.year}}},
              url = {{https://huggingface.co/{author}}},
              note = {{Dataset available at {hf_hub_repo_url(repo_id=repository, repo_type='dataset')}}}
            }}"""
    ).lstrip()
    dataset_infos = {
        "default": {
            "description": title,
            "citation": citation,
            "homepage": hf_hub_repo_url(repo_id=repository, repo_type='dataset'),
            "license": "other",
            "features": {
                "__key__": {
                    "dtype": "string",
                    "_type": "Value"
                },
                "__url__": {
                    "dtype": "string",
                    "_type": "Value"
                },
                "webp": {
                    "dtype": "image",
                    "_type": "Image"
                },
                "json": {
                    "id": {
                        "dtype": "int32",
                        "_type": "Value"
                    },
                    "width": {
                        "dtype": "int32",
                        "_type": "Value"
                    },
                    "height": {
                        "dtype": "int32",
                        "_type": "Value"
                    },
                    "class": {
                        "dtype": "string",
                        "_type": "Value"
                    }
                }
            },
            "builder_name": "webdataset",
            "config_name": "default",
            "version": {
                "version_str": f"{date.year}.{date.month}.{date.day}",
                "description": "Version based on release date (year.month.day)",
                "major": date.year,
                "minor": date.month,
                "patch": date.day
            }
        }
    }

    hf_fs = HfFileSystem(token=os.environ['HF_TOKEN_X'])
    hf_client = HfApi(token=os.environ['HF_TOKEN_X'])

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        hf_client.update_repo_settings(repo_id=repository, repo_type='dataset', gated='auto', private=True)
        # hf_client.update_repo_visibility(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        # attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        # attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('meta.json filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    global_seed(0)
    df = pd.DataFrame(records)
    df['r'] = df['id'].map(lambda x: random.random())
    df = df.sort_values(by=['r'], ascending=[True])
    del df['r']
    logging.info(f'Table to convert:\n{df}')

    if hf_fs.exists(f'datasets/{repository}/meta.json'):
        meta_info = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
        exist_ids = set(meta_info['exist_ids'])
        max_ids = meta_info['max_ids']
        split_infos = meta_info['split_infos']
        class_infos = meta_info['class_infos']
    else:
        exist_ids = set()
        max_ids = {}
        split_infos = {}
        class_infos = {}

    df_src = df
    df_src = df_src[~df_src['id'].isin(exist_ids)]
    df_src['r'] = df_src['id'].map(lambda x: random.random())
    df_src = df_src.sort_values(by=['split', 'r'], ascending=[False, True])
    del df_src['r']

    for split in ['train', 'test', 'val']:
        df_split = df_src[df_src['split'] == split]
        logging.info(f'Split {split!r} for syncing:\n{df_split}')

        batch_count = int(math.ceil(len(df_split) / batch_size))
        for batch_id in tqdm(range(batch_count), desc=f'Split {split!r}'):

            logging.info(f'Processing #{batch_id} for split {split!r} ...')
            df_batch = df_split[batch_size * batch_id: batch_size * (batch_id + 1)]
            if split not in split_infos:
                split_infos[split] = {'size': 0, 'image_count': 0}

            with TemporaryDirectory() as upload_dir:
                max_split_id = max_ids.get(split, 0)
                max_split_id += 1
                max_ids[split] = max_split_id
                shard_file = os.path.join(upload_dir, split, f'{max_split_id:05d}.tar')
                os.makedirs(os.path.dirname(shard_file), exist_ok=True)

                new_images_count = 0

                with tarfile.open(shard_file, 'a:') as tar:
                    lock = Lock()

                    def _fn_process(item):
                        nonlocal new_images_count
                        with TemporaryDirectory() as td:
                            bname = os.path.splitext(item['id'])[0].replace('/', '__')
                            dst_file = os.path.join(td, bname + '.webp')
                            image = load_image(os.path.join(image_folder, item['id']), force_background='white',
                                               mode='RGB')
                            if min(image.width, image.height) > min_size:
                                r = min(image.width, image.height) / min_size
                                new_width = int(image.width / r)
                                new_height = int(image.height / r)
                                logging.info(
                                    f'Resizing {item["id"]!r} from {image.size!r} to {(new_width, new_height)!r} ...')
                                image = image.resize((new_width, new_height))
                            image.save(dst_file)
                            dst_json_file = os.path.join(td, bname + '.json')

                            with open(dst_json_file, 'w') as f:
                                json.dump({
                                    'id': item['id'],
                                    'width': image.width,
                                    'height': image.height,
                                    'class': item['class'],
                                }, f)

                            with lock:
                                tar.add(dst_file, os.path.basename(dst_file))
                                tar.add(dst_json_file, os.path.basename(dst_json_file))
                                exist_ids.add(item['id'])
                                new_images_count += 1
                                split_infos[split]['image_count'] += 1
                                split_infos[split]['size'] += os.path.getsize(dst_file)

                                if item['class'] not in class_infos:
                                    class_infos[item['class']] = {'size': 0, 'image_count': 0}
                                class_infos[item['class']]['image_count'] += 1
                                class_infos[item['class']]['size'] += os.path.getsize(dst_file)

                    parallel_call(
                        df_batch.to_dict('records'),
                        fn=_fn_process,
                        max_workers=max_workers,
                        desc=f'Syncing for {split!r} #{max_split_id}'
                    )

                with open(os.path.join(upload_dir, 'meta.json'), 'w') as f:
                    json.dump({
                        'max_ids': max_ids,
                        'exist_ids': sorted(exist_ids),
                        'split_infos': split_infos,
                        'class_infos': class_infos,
                    }, f)

                with open(os.path.join(upload_dir, 'classes.json'), 'w') as f:
                    json.dump(sorted(class_infos.keys()), f)

                if pretrained_tag:
                    with open(os.path.join(upload_dir, 'pretrained_tag.json'), 'w') as f:
                        json.dump({'pretrained_tag': pretrained_tag}, f)

                with open(os.path.join(upload_dir, 'dataset_infos.json'), 'w') as f:
                    json.dump(dataset_infos, f, ensure_ascii=False, indent=4, sort_keys=True)

                md_file = os.path.join(upload_dir, 'README.md')
                with open(md_file, 'w') as f:
                    print(f'---', file=f)
                    print(f'license: {license}', file=f)
                    print(f'task_categories:', file=f)
                    print(f'- image-classification', file=f)
                    print(f'tags:', file=f)
                    print(f'- art', file=f)
                    print(f'- image', file=f)
                    print(f'- webdataset', file=f)
                    print(f'- datasets', file=f)
                    print(f'size_categories:', file=f)
                    print(f'- {number_to_tag(len(exist_ids))}', file=f)
                    print(f'---', file=f)
                    print(f'', file=f)

                    print(f'# {title}', file=f)
                    print(f'', file=f)
                    print(f'This is the webdataset dataset, '
                          f'containing {plural_word(len(exist_ids), "image")} in total.', file=f)
                    print(f'', file=f)
                    print(f'Images here are resized to `min(width, height) <= {min_size}`.', file=f)
                    print(f'', file=f)

                    print(f'## How to Use It', file=f)
                    print(f'', file=f)
                    print(f'```python\n'
                          f'from datasets import load_dataset\n'
                          f'\n'
                          f'dataset = load_dataset({repository!r})\n'
                          f'print(dataset["train"][0])\n'
                          f'```', file=f)
                    print(f'', file=f)

                    print(f'## Images', file=f)
                    print(f'', file=f)
                    print(f'{plural_word(len(exist_ids), "image")} in total.', file=f)
                    print(f'', file=f)
                    dx_rows = []
                    for s in ['train', 'test', 'val']:
                        if s not in split_infos:
                            ss, sc = 0, 0
                        else:
                            si = split_infos[s]
                            ss, sc = si['size'], si['image_count']
                        dx_rows.append({
                            'Split': s,
                            'Image Count': sc,
                            'Total Size': size_to_bytes_str(ss, sigfigs=3, system='si')
                        })
                    print(pd.DataFrame(dx_rows).to_markdown(index=False), file=f)
                    print(f'', file=f)

                    dx_rows = []
                    for s in sorted(class_infos.keys()):
                        if s not in class_infos:
                            ss, sc = 0, 0
                        else:
                            si = class_infos[s]
                            ss, sc = si['size'], si['image_count']
                        dx_rows.append({
                            'Class': s,
                            'Image Count': sc,
                            'Total Size': size_to_bytes_str(ss, sigfigs=3, system='si')
                        })
                    print(pd.DataFrame(dx_rows).to_markdown(index=False), file=f)
                    print(f'', file=f)

                    print('## Citation', file=f)
                    print(f'', file=f)
                    print(f'```\n'
                          f'{citation}\n'
                          f'```', file=f)
                    print(f'', file=f)

                upload_directory_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    local_directory=upload_dir,
                    path_in_repo='.',
                    message=f'Add pack #{max_split_id} for {split!r} with {plural_word(new_images_count, "image")}',
                    hf_token=os.environ['HF_TOKEN_X'],
                )


if __name__ == '__main__':
    cli()
