#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

from genericpath import exists
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
from torch.nn import  BCEWithLogitsLoss

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
    infere_grounding = 'grounding_inference' in complete_tasks

    if cfg.MODEL.GROUND_LAYERS_PER_TASK or cfg.MODEL.GROUND_LAYERS_TASK_LAYERS or cfg.MODEL.GROUND_LOSS_PER_TASK: 
        bce_logit = BCEWithLogitsLoss()
    
    for cur_iter, (inputs, labels, _, meta, texts) in enumerate(train_loader):
        # breakpoint()
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if 'prompt' not in key:
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if isinstance(val[i], (list,)):
                                val[i] = val[i][0].cuda(non_blocking=True)
                            else:
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

            for key, val in labels.items():
                if 'grounding' not in key:
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if isinstance(val[i], (list,)):
                                val[i] = val[i][0].cuda(non_blocking=True)
                            else:
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        labels[key] = val.cuda(non_blocking=True)
                elif key == 'grounding_inference':
                    labels[key] = val.cuda(non_blocking=True)
                else:
                    labels[key] = [val[0].cuda(non_blocking=True),val[1]]
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            with torch.autograd.set_detect_anomaly(True):
                faster_ftrs = meta["faster_features"] if cfg.FASTER.ENABLE else None
                keep_box = meta["keep_box"]

                preds = model(inputs, meta["boxes"], faster_ftrs, texts)
                # Explicitly declare reduction to mean and compute the loss for each task.
                loss = []
                for idx, task in enumerate(complete_tasks):
                    if task == 'actions':
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        loss.append(loss_fun(preds[task][0][keep_box], labels[task][keep_box]))    
                    elif task == 'tools':
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        loss.append(loss_fun(preds[task][0][keep_box], labels[task][keep_box].long()))    
                    elif task in ['phases','steps']:
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        indexes = np.unique(meta["boxes"][:,0].cpu(),return_index=True)[1]
                        loss.append(loss_fun(preds[task][0], labels[task][indexes].long()))
                    elif task == 'single_grounding':
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        grounding_preds,_ = preds[task][0]
                        grounding_labels = labels[task][0]
                        loss.append(loss_fun(grounding_preds,grounding_labels.long()))
                    elif task == 'combs_grounding' or task == 'perms_grounding' or task == 'action_grounding':
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        grounding_preds,_ = preds[task][0]
                        grounding_labels = labels[task][0]
                        loss.append(loss_fun(grounding_preds,grounding_labels.float()))
                    elif 'phrase' in task:
                        token_ids = meta['token_ids']
                        loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                        grounding_preds,_ = preds[task][0]
                        grounding_labels = labels[task][0]
                        scores = torch.bmm(token_ids.transpose(1,2).float(), grounding_preds)
                        scores /= (token_ids.sum(dim=1).unsqueeze(dim=2).repeat(1, 1, scores.shape[2]) + 0.0000001)
                        loss.append(loss_fun(scores,grounding_labels.float()))
                    elif task == 'grounding_inference':
                        if cfg.MODEL.GROUND_LAYERS_PER_TASK or cfg.MODEL.GROUND_LAYERS_TASK_LAYERS:
                            (tool_preds,action_preds),_ = preds[task][0]
                            grounding_labels = labels[task]
                            grounding_labels = grounding_labels.view(len(grounding_labels),cfg.MODEL.MAX_BBOX_NUM, 7, 16)
                            tool_labels,_ = torch.max(grounding_labels,dim=3)
                            action_labels,_ = torch.max(grounding_labels,dim=2)
                            tool_loss = bce_logit(tool_preds, tool_labels.float())
                            action_loss = bce_logit(action_preds, action_labels.float())
                            loss.append(tool_loss+action_loss)
                        elif cfg.MODEL.GROUND_LOSS_PER_TASK:
                            grounding_preds,_ = preds[task][0]
                            grounding_preds = grounding_preds.view(len(grounding_preds),cfg.MODEL.MAX_BBOX_NUM, 7, 16)
                            action_preds,_ = torch.max(grounding_preds,dim=2)
                            tool_preds,_ = torch.max(grounding_preds,dim=3)
                            grounding_labels = labels[task]
                            grounding_labels = grounding_labels.view(len(grounding_labels),cfg.MODEL.MAX_BBOX_NUM, 7, 16)
                            tool_labels,_ = torch.max(grounding_labels,dim=3)
                            action_labels,_ = torch.max(grounding_labels,dim=2)
                            tool_loss = bce_logit(tool_preds, tool_labels.float())
                            action_loss = bce_logit(action_preds, action_labels.float())
                            loss.append(tool_loss+action_loss)
                        else:
                            loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                            grounding_preds,_ = preds[task][0]
                            grounding_labels = labels[task]
                            loss.append(loss_fun(grounding_preds,grounding_labels.float()))
                    else:
                        raise ValueError(task)

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
    grounding = any('grounding' in task for task in complete_tasks)
    phrase = any('phrase' in task for task in complete_tasks)
    infere_grounding = 'grounding_inference' in complete_tasks
    
    for cur_iter, (inputs, labels, _, meta, texts) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if 'prompt' not in key:
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if isinstance(val[i], (list,)):
                                val[i] = val[i][0].cuda(non_blocking=True)
                            else:
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            
            for key, val in labels.items():
                if 'grounding' not in key:
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            if isinstance(val[i], (list,)):
                                val[i] = val[i][0].cuda(non_blocking=True)
                            else:
                                val[i] = val[i].cuda(non_blocking=True)
                    else:
                        labels[key] = val.cuda(non_blocking=True)
                else:
                    labels[key] = [val[0].cuda(non_blocking=True),val[1]]
        # breakpoint()
        val_meter.data_toc()
        faster_ftrs = meta["faster_features"] if cfg.FASTER.ENABLE else None
        preds = model(inputs, meta["boxes"], faster_ftrs, texts)
        # breakpoint()
        
        keep_box = meta["keep_box"]
        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]
        image_names = meta["img_names"]
        if cfg.NUM_GPUS:
            new_preds = {}
            for task in complete_tasks:
                if 'grounding' in task:
                    # breakpoint()
                    if cfg.MODEL.GROUND_LAYERS_PER_TASK or cfg.MODEL.GROUND_LAYERS_TASK_LAYERS:
                        new_preds[task] = ((preds[task][0][0][0].cpu(),preds[task][0][0][1].cpu()),preds[task][0][1].cpu())
                    else:
                        new_preds[task] = (preds[task][0][0].cpu(),preds[task][0][1].cpu())
                else:
                    new_preds[task] = preds[task][0].cpu()
            preds = new_preds
            # preds = {task: preds[task][0].cpu() if 'grounding' not in task else (preds[task][0][0].cpu(),preds[task][0][1].cpu()) for task in complete_tasks}
            ori_boxes = ori_boxes.cpu()
            metadata = metadata.cpu()
            keep_box = keep_box.cpu()
            image_names = image_names.cpu()
            if cfg.NUM_GPUS > 1:
                preds_gather = {}
                for pred in preds:
                    preds_gather[pred] = torch.cat(du.all_gather_unaligned(preds[pred]), dim=0)
                preds = preds_gather.copy()
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                image_names = torch.cat(du.all_gather_unaligned(image_names), dim=0)
                keep_box = torch.cat(du.all_gather_unaligned(keep_box), dim=0)
        # breakpoint()
        val_meter.iter_toc()
        epoch_names_detect, epoch_bboxes = [], []
        for image_name, batch_box in zip(image_names[keep_box], ori_boxes[keep_box].cpu().tolist()):
            epoch_names_detect.append(''.join(map(chr,image_name.cpu().tolist())))
            epoch_bboxes.append([batch_box[j] for j in [1, 2, 3, 4]])        

        # Images names phases/steps
        epoch_names = []
        for image_name in image_names:
            epoch_names.append(''.join(map(chr,image_name.cpu().tolist())))   
        
        epoch_names = list(np.unique(epoch_names))   
        # breakpoint()
        # Update and log stats.
        if grounding:
            grounding_task = complete_tasks[-1]
            gr_names = []
            gr_ids = set([])
            for imid,image_name in enumerate(image_names):
                if int(ori_boxes[imid,0]) not in gr_ids:
                    gr_names.append(''.join(map(chr,image_name.cpu().tolist())))
                    gr_ids.add(int(ori_boxes[imid,0]))
            
            grounding_preds,vis_att_mask = preds[grounding_task] 
            if phrase:
                token_ids = meta['token_ids'].cpu()
                scores = torch.bmm(token_ids.transpose(1,2).float(), grounding_preds)
                scores /= (token_ids.sum(dim=1).unsqueeze(dim=2).repeat(1, 1, scores.shape[2]) + 0.0000001)
                grounding_preds = scores
            if not infere_grounding:
                _,gt_boxes = labels[grounding_task]
                val_meter.update_grounding(grounding_preds.cpu(), vis_att_mask.cpu(), ori_boxes.cpu(), gt_boxes, gr_names, texts)
            else:
                # breakpoint()
                if cfg.MODEL.GROUND_LAYERS_PER_TASK or cfg.MODEL.GROUND_LAYERS_TASK_LAYERS:
                    tool_preds,action_preds = grounding_preds
                    tool_preds = softmax(tool_preds[vis_att_mask==1],dim=-1)
                    action_preds = sigmoid(action_preds[vis_att_mask==1])
                    preds['tools'] = tool_preds
                    preds['actions'] = action_preds
                    del preds['grounding_inference']
                    val_meter.update_stats(preds, keep_box, epoch_bboxes, epoch_names_detect, epoch_names)
                else:
                    grounding_preds = torch.sigmoid(grounding_preds.view(len(inputs[0]),cfg.MODEL.MAX_BBOX_NUM, 7, 16))
                    grounding_act_preds = torch.max(grounding_preds,dim=2)[0][vis_att_mask==1]
                    grounding_tool_preds = torch.max(grounding_preds,dim=3)[0][vis_att_mask==1]
                    preds['tools'] = grounding_tool_preds
                    preds['actions'] = grounding_act_preds
                    del preds['grounding_inference']
                    val_meter.update_stats(preds, keep_box, epoch_bboxes, epoch_names_detect, epoch_names)
        
        if any('grounding' not in task for task in complete_tasks):
            val_meter.update_stats(preds, keep_box, epoch_bboxes, epoch_names_detect, epoch_names)

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


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # TODO Si no corre, quitarlo
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
            
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    if any('grounding' in task for task in cfg.TASKS.TASKS) and cfg.TRAIN.PRETRAIN in ['cross','full'] and start_epoch==0:
        cu.load_ground_checkpoint(cfg,model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )
    
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    # Perform final test
    if cfg.TEST.ENABLE:
        logger.info("Evaluating epoch: {}".format(start_epoch + 1))
        map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, start_epoch, cfg)
        return
    else:
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_map = {task: 0 for task in (complete_tasks if not 'grounding_inference' in complete_tasks else ['tools','actions'])}
    best_mean_map = 0
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )
            
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
        if os.path.exists(del_fil):
            os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                # Save best results
                if mean_map > best_mean_map:
                    best_mean_map = mean_map
                    logger.info("Best mean map at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in (complete_tasks if not 'grounding_inference' in complete_tasks else ['tools','actions']):
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in (complete_tasks if not 'grounding_inference' in complete_tasks else ['tools','actions']):
                    if map_task[task] > best_task_map[task]:
                        best_task_map[task] = map_task[task]
                        logger.info("Best {} map at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

