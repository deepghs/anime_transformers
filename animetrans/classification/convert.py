import glob
import mimetypes
import os
from functools import partial

import click
import pandas as pd
from ditk import logging
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
def id2datasets(image_folder: str, min_image_class: int):
    logging.try_init_root(logging.INFO)

    mimetypes.add_type('image/webp', '.webp')
    try:
        import pillow_avif
    except (ImportError, ModuleNotFoundError) as err:
        logging.warning(f'Pillow Avif not launched - {err!r}.')
    else:
        logging.info('Pillow Avif launched.')

    d_classes = {}
    for name in tqdm(os.listdir(image_folder), desc='Scanning classes'):
        if os.path.isdir(os.path.join(image_folder, name)):
            d_classes[name] = []
            for file in tqdm(glob.glob(os.path.join(image_folder, name, '**', '*'), recursive=True),
                             desc=f'Scanning class {name!r}'):
                mimetype, _ = mimetypes.guess_type(file)
                if mimetype and mimetype.startswith('image/'):
                    d_classes[name].append(hf_normpath(os.path.relpath(file, image_folder)))

    d_classes = {name: value for name, value in d_classes.items() if len(value) >= min_image_class}
    classes = natsorted(d_classes.keys())
    df_classes = pd.DataFrame([
        {'class': name, 'count': len(d_classes)}
        for name in classes
    ])
    logging.info(f'Classes analysis:\n{df_classes}')


if __name__ == '__main__':
    cli()
