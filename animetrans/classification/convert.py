import glob
import mimetypes
import os
from functools import partial

import click
import pandas as pd
from ditk import logging
from hbutils.encoding import int_hash
from hfutils.utils import hf_normpath
from natsort import natsorted
from tqdm import tqdm

from ..utils import GLOBAL_CONTEXT_SETTINGS, print_version


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
def id2datasets(image_folder: str, min_image_class: int, eval_percentile: int, test_percentile: int):
    logging.try_init_root(logging.INFO)

    mimetypes.add_type('image/webp', '.webp')
    try:
        import pillow_avif
    except (ImportError, ModuleNotFoundError) as err:
        logging.warning(f'Pillow Avif not launched - {err!r}.')
    else:
        logging.info('Pillow Avif launched.')

    d_class_splits = {}
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


if __name__ == '__main__':
    cli()
