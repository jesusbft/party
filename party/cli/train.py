#
# Copyright 2022 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
party.cli.train
~~~~~~~~~~~~~~~~~~

Command line driver for recognition training.
"""
import logging

import click
from threadpoolctl import threadpool_limits

from party.default_specs import RECOGNITION_HYPER_PARAMS

from .util import _expand_gt, _validate_manifests, message, to_ptl_device

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('avg_ckpts')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='average.ckpt', help='Averaged model file path.')
@click.option('-n', '--num-checkpoints', show_default=True, default=5, type=click.IntRange(2), help='Number of final checkpoints to average.')
@click.argument('input', nargs=1, type=click.Path(exists=True, dir_okay=True, file_okay=False))
def avg_ckpts(ctx, output, num_checkpoints, input):
    """
    Averages n-last checkpoints in input directory.
    """
    import glob
    import torch
    import collections
    ckpts = sorted(glob.glob(f'{input}/checkpoint_*-*.ckpt'))
    message(f'Found {len(ckpts)} checkpoints in {input}')
    ckpts = ckpts[-num_checkpoints:]
    if len(ckpts) < num_checkpoints:
        raise click.BadParameter(f'Less checkpoints found than requested for averaging ({len(ckpts)} < {num_checkpoints})', param_hint='input')
    message(f'Averaging {ckpts}')

    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(ckpts)

    for fpath in ckpts:
        with open(fpath, "rb") as f:
            state = torch.load(f, map_location=(lambda s, _: torch.serialization.default_restore_location(s, 'cpu')),)
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['state_dict']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state['state_dict'] = averaged_params
    message(f'Writing averaged checkpoint to {output}')
    torch.save(new_state, output)


@click.command('train')
@click.pass_context
@click.option('-i', '--load', default=None, type=click.Path(exists=True), help='Checkpoint to load')
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('--line-height', show_default=True, type=click.INT, default=RECOGNITION_HYPER_PARAMS['height'],
              help='Input line height to network after scaling')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-F', '--freq', show_default=True, default=RECOGNITION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'fixed']),
              help='Stop condition for training. Set to `early` for early stooping or `fixed` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequency) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('--optimizer',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'AdamW',
                                 'SGD',
                                 'RMSprop']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=RECOGNITION_HYPER_PARAMS['lr'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=RECOGNITION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, type=float,
              default=RECOGNITION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--warmup', show_default=True, type=int,
              default=RECOGNITION_HYPER_PARAMS['warmup'], help='Number of steps to ramp up to `lrate` initial learning rate.')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=RECOGNITION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('--cos-min-lr',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_min_lr'],
              help='Minimal final learning rate for cosine LR scheduler.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=RECOGNITION_HYPER_PARAMS['normalization'], help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace', show_default=True,
              default=RECOGNITION_HYPER_PARAMS['normalize_whitespace'], help='Normalizes unicode whitespace')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, type=click.IntRange(1), help='Number of worker processes.')
@click.option('--threads', show_default=True, default=1, type=click.IntRange(1), help='Maximum size of OpenMP/BLAS thread pool.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, load, batch_size, line_height, output, freq, quit, epochs,
          min_epochs, lag, min_delta, optimizer, lrate, momentum, weight_decay,
          warmup, schedule, gamma, step_size, sched_patience,
          cos_max, cos_min_lr, normalization, normalize_whitespace, reorder,
          base_dir, training_files, evaluation_files, workers, threads,
          augment, ground_truth):
    """
    Trains a model from image-text pairs.
    """
    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations  # NOQA
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    import torch

    from party.dataset import TextLineDataModule
    from party.model import RecognitionModel

    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, RichProgressBar

    torch.set_float32_matmul_precision('medium')

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update({'freq': freq,
                         'height': line_height,
                         'batch_size': batch_size,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lr': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'warmup': warmup,
                         'schedule': schedule,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'cos_min_lr': cos_min_lr,
                         'normalization': normalization,
                         'normalize_whitespace': normalize_whitespace,
                         'augment': augment,
                         })

    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    data_module = TextLineDataModule(training_data=ground_truth,
                                     evaluation_data=evaluation_files,
                                     height=hyper_params['height'],
                                     augmentation=augment,
                                     batch_size=batch_size,
                                     num_workers=workers,
                                     reorder=reorder,
                                     normalization=hyper_params['normalization'],
                                     normalize_whitespace=hyper_params['normalize_whitespace'])

    if load:
        message('Loading model.')
        model = RecognitionModel.load_from_checkpoint(load,
                                                      num_classes=data_module.num_classes,
                                                      map_location=torch.device('cpu'),
                                                      pad_id=data_module.pad_id,
                                                      sos_id=data_module.sos_id,
                                                      eos_id=data_module.eos_id,
                                                      **hyper_params)

    else:
        message('Initializing model.')
        model = RecognitionModel(**hyper_params,
                                 num_classes=data_module.num_classes,
                                 pad_id=data_module.pad_id,
                                 sos_id=data_module.sos_id,
                                 eos_id=data_module.eos_id)

    cbs = [RichModelSummary(max_depth=2)]

    checkpoint_callback = ModelCheckpoint(dirpath=output,
                                          save_top_k=10,
                                          monitor='global_step',
                                          mode='max',
                                          auto_insert_metric_name=False,
                                          filename='checkpoint_{epoch:02d}-{val_metric:.4f}')

    cbs.append(checkpoint_callback)
    if not ctx.meta['verbose']:
        cbs.append(RichProgressBar(leave=True))

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision=ctx.meta['precision'],
                      max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'fixed' else -1,
                      min_epochs=hyper_params['min_epochs'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=ctx.meta['deterministic'],
                      enable_model_summary=False,
                      accumulate_grad_batches=4,
                      callbacks=cbs,
                      **val_check_interval)

    with threadpool_limits(limits=threads):
        trainer.fit(model, data_module)

    if model.best_epoch == -1:
        logger.warning('Model did not improve during training.')
        ctx.exit(1)

    if not model.current_epoch:
        logger.warning('Training aborted before end of first epoch.')
        ctx.exit(1)

    print(f'Best model {checkpoint_callback.best_model_path}')
