#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import os
from re import X
import tarfile
import tempfile
import traceback
import torch
import torch.nn as nn
from detectron2.layers import ROIAlign


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


class TransformerRoIHead(nn.Module):
    """
    Box classification head in TAPIR. 
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        
        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
            
        
        # Feature Vector dimension from faster is 1024
        dim_faster = 1024 
        # Feature Vector dimension from deformable detr is 1024
        dim_detr = 256
             
        if self.cfg.FASTER.ENABLE:
            self.dim_add = 1024
            dim_out = self.dim_add + 768
            
        else: 
            self.dim_add = 0
            dim_out = self.dim_add + 1024
        
        
        if cfg.FASTER.DETR:
            # We redimension deformable detr features to the same as in faster r-cnn
            self.mlp = nn.Sequential(nn.Linear(dim_detr, dim_faster, bias=False),
                                    nn.BatchNorm1d(dim_faster))
        elif not self.cfg.FASTER.ENABLE:
            self.mlp = nn.Sequential(nn.Linear(768, dim_out, bias=False),
                                    nn.BatchNorm1d(dim_out))
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, num_classes, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, features=None):
        x = inputs.mean(1)
        x_boxes = torch.zeros(len(bboxes), x.shape[1], device = inputs.device, requires_grad=True)
        for i in range(len(inputs)):
            x_boxes[bboxes[:,0] == i].copy_(x[i])
            x[i].detach()
        
        if features is not None:
            features = features[:,1:]
            if self.cfg.FASTER.DETR:
                features = self.mlp(features)
            x = torch.cat([x_boxes, features], dim=1)

        elif features is None and self.cfg.FASTER.ENABLE:
            features = torch.zeros(x_boxes.shape[0], self.dim_add, device = inputs.device, requires_grad=True)
            x = torch.cat([x_boxes, features], dim=1)

        elif features is None and not self.cfg.FASTER.ENABLE:
            x = self.mlp(x)
            x_boxes = torch.zeros(len(bboxes), x.shape[1], device = inputs.device, requires_grad=True)
            for i in range(len(inputs)):
                x_boxes[bboxes[:,0] == i].copy_(x[i])
                x[i].detach()
            x = x_boxes
        
        x = self.projection(x)
        if self.training and self.act_func == "sigmoid" or not self.training:
            x = self.act(x)

        return x
        

class TransformerGroundHead(nn.Module):
    """
    Box classification head in TAPIR. 
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        
        super(TransformerGroundHead, self).__init__()
        self.cfg = cfg
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
            
        
        # Feature Vector dimension from faster is 1024
        dim_faster = 1024 
        # Feature Vector dimension from deformable detr is 1024
        dim_detr = 256
        self.trans = False
        self.deep = cfg.MODEL.DEEP_SUPERVISION
             
        if self.cfg.FASTER.ENABLE:
            self.dim_add = 1024
            dim_out = self.dim_add + 768
            
        else: 
            self.dim_add = 0
            dim_out = self.dim_add + 1024
        
        
        if cfg.FASTER.DETR:
            # We redimension deformable detr features to the same as in faster r-cnn
            if cfg.MODEL.MLP:
                self.trans = True
                self.mlp = nn.Sequential(nn.Linear(dim_detr,768),
                                         nn.ReLU(),
                                         nn.Linear(768,768))

                self.transform = nn.Sequential(nn.Linear(768,768),
                                               nn.ReLU(),
                                               nn.Linear(768,768))
                dim_out = 768*2
            else:
                self.mlp = nn.Sequential(nn.Linear(dim_detr, dim_faster, bias=False),
                                         nn.BatchNorm1d(dim_faster))
            
            if self.deep:
                self.deep_trans = nn.Sequential(nn.Linear(768,768),
                                                nn.ReLU(),
                                                nn.Linear(768,768))

        elif not self.cfg.FASTER.ENABLE:
            self.mlp = nn.Sequential(nn.Linear(768, dim_out, bias=False),
                                    nn.BatchNorm1d(dim_out))
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, 768, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, features=None):
        x = inputs.mean(1)
        ret_x = x
        if self.deep:
            ret_x = self.deep_trans(x)
        if self.trans:
            x = self.transform(x)
        
        x_boxes = torch.zeros(len(bboxes), x.shape[1], device = inputs.device, requires_grad=True)
        # bidx = torch.zeros(len(bboxes), device = inputs.device, requires_grad=False)
        for i in range(len(inputs)):
            # bidx[bboxes[:,0] == i] = i
            x_boxes[bboxes[:,0] == i].copy_(x[i])
            x[i].detach()
        
        if features is not None:
            features = features[:,1:]
            if self.cfg.FASTER.DETR:
                features = self.mlp(features)
            x = torch.cat([x_boxes, features], dim=1)

        elif features is None and self.cfg.FASTER.ENABLE:
            features = torch.zeros(x_boxes.shape[0], self.dim_add, device = inputs.device, requires_grad=True)
            x = torch.cat([x_boxes, features], dim=1)

        elif features is None and not self.cfg.FASTER.ENABLE:
            x = self.mlp(x)
            x_boxes = torch.zeros(len(bboxes), x.shape[1], device = inputs.device, requires_grad=True)
            for i in range(len(inputs)):
                x_boxes[bboxes[:,0] == i].copy_(x[i])
                x[i].detach()
            x = x_boxes
        
        x = self.projection(x)

        vis_output = []
        att_mask = torch.zeros((len(inputs),self.cfg.MODEL.MAX_BBOX_NUM), device=inputs.device)
        for i in range(len(inputs)):
            log_ind_mask = (bboxes[:,0] == i)
            num_bboxes = len(bboxes[log_ind_mask])
            if num_bboxes==self.cfg.MODEL.MAX_BBOX_NUM:
                vis_output.append(x[log_ind_mask])
                att_mask[i][:] = 1
            else:
                padding_tensor = torch.zeros((self.cfg.MODEL.MAX_BBOX_NUM-num_bboxes,x.shape[1]), device=inputs.device)
                vis_output.append(torch.cat((x[log_ind_mask],padding_tensor),dim=0))
                att_mask[i][:num_bboxes] = 1
        
        vis_output = torch.cat(vis_output,dim=0).view((len(inputs),self.cfg.MODEL.MAX_BBOX_NUM,x.shape[1]))

        return vis_output,att_mask,ret_x


class TextEncoder(nn.Module):
    """
        Bert language processor
    """

    def __init__(self, cfg):
        super(TextEncoder, self).__init__()

        if cfg.MODEL.TEXT_ENCODER == 'bert':
            from transformers import BertModel, BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        elif cfg.MODEL.TEXT_ENCODER == 'roberta':
            from transformers import RobertaModel, RobertaTokenizer

            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert = RobertaModel.from_pretrained('roberta-base')

        elif cfg.MODEL.TEXT_ENCODER == 'biobert':
            from transformers import AutoModel, AutoConfig, AutoTokenizer

            config = AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", use_fast=False)
            self.bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2",
                                            from_tf=False, config=config, cache_dir=None)
            # cfg.MODEL.MAX_SEQUENCE_LENGTH += 2
                                            
        elif cfg.MODEL.TEXT_ENCODER == 'clinicbert':
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            # cfg.MODEL.MAX_SEQUENCE_LENGTH += 2
            
        else:
            raise ValueError(f'The encoder {cfg.MODEL.TEXT_ENCODER} is not recognized')

        self.cfg = cfg
    
    def forward(self, texts):

        try:
            tokens = self.tokenizer(texts, padding='max_length', max_length = self.cfg.MODEL.MAX_SEQUENCE_LENGTH, return_tensors="pt")
        except ValueError:
            traceback.print_exc()
            breakpoint()
            
        tokens_ids = tokens['input_ids'].cuda()
        att_masks = tokens['attention_mask'].cuda()

        hidden,cls_token = self.bert(input_ids=tokens_ids, 
                    attention_mask=att_masks,
                    return_dict=False)
        
        return hidden, att_masks, cls_token


class CrossEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        from .png.modeling import VisualFeatEncoder, Cross_Layer, BertLayer, BertPooler
        from .png.file_utils import cached_path
        from .png.bert_config import BertConfig

        archive_file = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=None)
        except EnvironmentError:
            traceback.print_exc()
            breakpoint()

        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()

            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, 'bert_config.json')
        config = BertConfig.from_json_file(config_file)

        self.visn_fc = VisualFeatEncoder(config)
        self.cross_layers_num = 5
        self.single_layers_num = 3

        self.cross_layers = nn.ModuleList(
            [Cross_Layer(config) for _ in range(self.cross_layers_num)]
        )
        self.single_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.single_layers_num)]
        )

        self.pooler = BertPooler(config)

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None):

        visn_feats = self.visn_fc(visn_feats)

        extended_attention_mask = lang_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_visual_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0


        for layer_module in self.single_layers:
            visn_feats = layer_module(visn_feats, extended_visual_attention_mask)
        ret_visn_feats = visn_feats
            
        # cross-modality
        for layer_module in self.cross_layers:
            lang_feats, visn_feats = layer_module(lang_feats, extended_attention_mask,
                                                  visn_feats, extended_visual_attention_mask)
        
        pooled_output = self.pooler(lang_feats)

        return lang_feats, visn_feats, pooled_output, ret_visn_feats
        
