#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

from genericpath import exists
import random
import numpy as np
import shutil
import os
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import EpochTimer, SurgeryMeter
from slowfast.utils.multigrid import MultigridSchedule
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.functional import softmax, sigmoid
from torch.nn import  BCEWithLogitsLoss, CrossEntropyLoss

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py

    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    complete_tasks = cfg.TASKS.TASKS
    complete_loss_funs = cfg.TASKS.LOSS_FUNC
    
    for cur_iter, (inputs, labels, names, texts) in enumerate(train_loader):
        # breakpoint()
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            
            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda(non_blocking=True)
            else:
                labels = labels.cuda(non_blocking=True)
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        # breakpoint()
        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs, texts=texts)[complete_tasks[0]][0]
            # Explicitly declare reduction to mean and compute the loss for each task.
            loss = []
            try:
                loss_fun = losses.get_loss_func(complete_loss_funs[0])(reduction="mean")
                loss.append(loss_fun(preds,labels.float()))
            except:
                breakpoint()

        if len(complete_tasks) >1:
            final_loss = losses.compute_weighted_loss(loss, cfg.TASKS.LOSS_WEIGHTS)
        else:
            final_loss = loss[0]
            
        # check Nan Loss.
        misc.check_nan_losses(final_loss)

        # Perform the backward pass.
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.NUM_GPUS > 1:
            final_loss = du.all_reduce([final_loss])[0]
        final_loss = final_loss.item()

        # Update and log stats.
        train_meter.update_stats(None, None, None, None, None, final_loss, loss, lr)
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    
    for cur_iter, (inputs, _, names, texts) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

        val_meter.data_toc()
        preds = model(inputs, texts=texts[0])[complete_tasks[0]][0]

        val_meter.iter_toc()     
        epoch_names = set(names)  

        # Update and log stats.
        val_meter.update_stats({'phases': softmax(preds,dim=1).cpu()}, [True]*len(preds), [[0,0,1,1]]*len(preds), names, epoch_names)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)

    if cfg.NUM_GPUS > 1:
        if du.is_master_proc():
            task_map, mean_map, out_files = val_meter.finalize_metrics()
        else:
            task_map, mean_map, out_files =  [0, 0, 0]
        torch.distributed.barrier()
    else:
        task_map, mean_map, out_files = val_meter.finalize_metrics()
    val_meter.reset()

    return task_map, mean_map, out_files