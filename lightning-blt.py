import time
from pathlib import Path

import torch
from torch.nn.modules.loss import NLLLoss
from pytorch_dataset_loaders.optimizers import Adam
from pytorch_dataset_loaders.pytorch_transforms import Transforms
from pytorch_dataset_loaders.pytorch_datasets import GenericDataModule, Ecoset
from pytorch_dataset_loaders.schedulers import ReduceLROnPlateau
from pytorch_dataset_loaders.lightning_trainer import LightningModel, LightningSpecialTrain
import pytorch_dataset_loaders.metrics as metrics_utils
from pytorch_dataset_loaders.pytorch_callbacks import CustomProgressBar
from pytorch_dataset_loaders.pytorch_loggers import CustomNPZLogger

from blt_net.blt_net import BLT_net


def main():

    # Get all the hyperparameters needed as a dictionary
    hyperparams = set_hyperparameters()
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
                                 prefetch_factor_test=hyperparams['prefetch_factor_val_test'],
                                 train_transformations=train_transformations,
                                 val_transformations=val_test_transformations,
                                 test_transofrmations=val_test_transformations)

    model = BLT_net(n_blocks=3,
                    n_layers=1,
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
        save_dir=hyperparams['log_path'], name=hyperparams['dataset_name'], timesteps=hyperparams['timesteps'])

    # Now we have everything (except the logger) to start the training
    lightning_special_train = LightningSpecialTrain(blt_data, lightning_model, ckpt_path=hyperparams['ckpt_path'],
                                                    epochs=hyperparams['n_epochs'],
                                                    accelerator=hyperparams['accelerator'],
                                                    callbacks=[
                                                        progress.progress_bar],
                                                    logger=npz_logger,
                                                    devices=hyperparams['devices'],
                                                    strategy="ddp_find_unused_parameters_true")

    lightning_special_train._train_model()

    # Save metrics to json file
    metrics = metrics_utils.extract_metrics_from_model(lightning_model)
    experiment_name = time.strftime("%Y%m%d-%H%M%S")
    metrics_utils.save_metrics_to_file(metrics, hyperparams['log_path'], experiment_name)


def set_hyperparameters():

    hyperparams = {

        'dataset_name': 'miniecoset_',
        'image_size': 64,
        'dataset_dir': '/share/klab/datasets/',
        'log_path': 'lightning_logs/',
        'ckpt_path': None,
        'accelerator': 'gpu',
        'devices': '-1',
        'augment': {'trivialaug', 'normalize'},
        'model': 'BLT_net',  # model to be used
        'identifier': '1',  # identifier in case we run multiple versions of the net
        'use_blt_loss': True,
        'timesteps': 8,  # number of timesteps to unroll the RNN
        'lateral_connections': 1,  # if lateral connections should exist throughout the network
        'topdown_connections': 1,  # if topdown connections should exist throughout the network
        # 'additive' or 'multiplicative' interaction with bottom-up flow
        'LT_interaction': 'additive',
        'LT_position': 'all',  # 'all' = everywhere, 'last' = at the GAP layer
        'classifier_bias': 0,  # if the classifier layer should have a bias parameter
        'norm_type': 'LN',  # which norm to use - 'LN', 'None'
        'learning_rate': 0.001,
        'batch_size_train': 1024,
        # issue: {a single A100 could not handle 1k batch size for val}
        'batch_size_val_test': 250,
        'n_epochs': 100,
        'train_img_mean_channels': [0.4987, 0.4702, 0.4050],
        'train_img_std_channels': [0.2711, 0.2635, 0.2810],
        'automatic_optimization': False,
        'num_workers_train': 10,  # number of cpu workers processing the batches
        # number of batches kept in memory by each worker (providing quick access for the gpu)
        'prefetch_factor_train': 4,
        'num_workers_val_test': 3,  # do not need lots of workers for val/test
        'prefetch_factor_val_test': 4,
        'pin_memory': True,  # if the data should be pinned to memory
        # use automatic mixed precision during training - forward pass .half(), backward full
        'use_amp': True,
        'save_logs': 10,  # after how many epochs should we save a copy of the logs
        'save_net': 30  # after how many epochs should we save a copy of the net
    }
    if hyperparams['timesteps'] == 1:  # if only 1 timestep requested, then no LT
        hyperparams['lateral_connections'] = 0
        hyperparams['topdown_connections'] = 0
        hyperparams['LT_interaction'] = None
        hyperparams['LT_position'] = None

    return hyperparams


if __name__ == '__main__':
    main()
