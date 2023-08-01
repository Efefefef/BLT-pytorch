import time
import json
from pathlib import Path
import os

import torch
from torch.nn.modules.loss import NLLLoss
from pytorch_dataset_loaders.optimizers import Adam
from pytorch_dataset_loaders.pytorch_transforms import Transforms
from pytorch_dataset_loaders.pytorch_datasets import GenericDataModule, Ecoset
from pytorch_dataset_loaders.schedulers import ReduceLROnPlateau
from pytorch_dataset_loaders.lightning_trainer import LightningModel, LightningModelTrain
from pytorch_dataset_loaders.pytorch_callbacks import CustomProgressBar, CheckpointModel
from pytorch_dataset_loaders.pytorch_loggers import CustomNPZLogger

from src.blt_net import BLT_net


def extract_metrics_from_model(lightning_model: LightningModel):
    # to account for the last additional validation run done by the trainer
    extra_validation_runs = 1
    metrics = {
        'train_losses_epoch': lightning_model.train_losses_epoch,
        'train_losses_step': lightning_model.train_losses_step,
        'train_accuracies_epoch': lightning_model.train_accuracies_epoch,

        'val_losses_step': lightning_model.val_losses_step,
        'val_losses_epoch': lightning_model.val_losses_epoch[:-extra_validation_runs],
        'val_accuracies_step': lightning_model.val_accuracies_step,
        'val_accuracies_epoch': lightning_model.val_accuracies_epoch[:-extra_validation_runs],

        'time_per_epoch': lightning_model.epoch_times
    }
    return metrics


def save_metrics_to_file(metrics, path, filename):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{filename}.json'), 'w') as file:
        json.dump(metrics, file, indent=4)


def main():

    with open('hyperparams.json', 'r') as file:
        hyperparams = json.load(file)

    experiment_name = (
        f'{hyperparams["n_blocks"]}-blocks_'
        f'{hyperparams["n_layers"]}-layers_'
        f'{hyperparams["timesteps"]}-timesteps'
        f'_{time.strftime("%Y%m%d-%H%M%S")}'
    )

    Path(hyperparams['log_path']).mkdir(exist_ok=True)
    # Get the dataset path
    dataset_path = hyperparams['dataset_dir'] + \
        hyperparams['dataset_name'] + str(hyperparams['image_size']) + 'px.h5'

    # Get the data transformations
    train_transformations = Transforms(
        aug_str=hyperparams['augment'], mean=hyperparams['train_img_mean_channels'], std=hyperparams['train_img_std_channels'])
    val_test_transformations = Transforms(
        aug_str=['normalize'], mean=hyperparams['train_img_mean_channels'], std=hyperparams['train_img_std_channels'])

    # Create the lightning datamodule
    dataset_train = Ecoset('train', dataset_path,
                           transform=train_transformations.get_transform())
    dataset_val = Ecoset('val', dataset_path,
                         transform=val_test_transformations.get_transform())
    dataset_test = Ecoset('test', dataset_path,
                          transform=val_test_transformations.get_transform())
    blt_data = GenericDataModule(dataset_train, dataset_test, dataset_val,
                                 batch_size_train=hyperparams['batch_size_train'],
                                 batch_size_val=hyperparams['batch_size_val_test'],
                                 batch_size_test=hyperparams['batch_size_val_test'],
                                 num_workers_train=hyperparams['num_workers_train'],
                                 num_workers_val=hyperparams['num_workers_val_test'],
                                 num_workers_test=hyperparams['num_workers_val_test'],
                                 prefetch_factor_train=hyperparams['prefetch_factor_train'],
                                 prefetch_factor_val=hyperparams['prefetch_factor_val_test'],
                                 prefetch_factor_test=hyperparams['prefetch_factor_val_test'])

    model = BLT_net(n_blocks=hyperparams['n_blocks'],
                    n_layers=hyperparams['n_layers'],
                    is_lateral_enabled=hyperparams['lateral_connections'],
                    is_topdown_enabled=hyperparams['topdown_connections'],
                    LT_interaction=hyperparams['LT_interaction'],
                    timesteps=hyperparams['timesteps'])

    scalar = torch.cuda.amp.GradScaler(hyperparams['use_amp'])
    optimizer = Adam(lr=hyperparams['learning_rate'])
    loss = NLLLoss()
    progress = CustomProgressBar()
    scheduler = ReduceLROnPlateau()
    lightning_model = LightningModel(model, loss, optimizer, scheduler=scheduler, automatic_optimization=hyperparams[
                                     'automatic_optimization'], scalar=scalar, use_blt_loss=hyperparams['use_blt_loss'], timesteps=hyperparams['timesteps'])

    # Initialize our custom logger
    npz_logger = CustomNPZLogger(
        save_dir=hyperparams['log_path'],
        experiment_name=experiment_name,
        timesteps=hyperparams['timesteps'])

    ckpt_path = os.path.join(hyperparams['log_path'], experiment_name)
    modelcheckpoint = CheckpointModel(save_dir=hyperparams['log_path'], filename=experiment_name, every_n_epochs=10)

    lightning_special_train = LightningModelTrain(blt_data,
                                                    lightning_model,
                                                    ckpt_path=ckpt_path,
                                                    epochs=hyperparams['n_epochs'],
                                                    accelerator=hyperparams['accelerator'],
                                                    callbacks=[
                                                        progress.progress_bar,
                                                        modelcheckpoint.checkpoint_callback
                                                        ],
                                                    logger=npz_logger,
                                                    devices=hyperparams['devices'],
                                                    strategy="ddp_find_unused_parameters_true")

    lightning_special_train._train_model()

    # Save metrics to json file (in addition to the npz logging done by the lightning logger)
    metrics = extract_metrics_from_model(lightning_model)
    save_metrics_to_file(metrics, hyperparams['log_path'], experiment_name)


if __name__ == '__main__':
    main()
