# Experiment setup
FOLD="2" 
TEXT_ENCODER="biobert"
TASK="VARIS_GROUND"
CHECKPOINT="/media/SSD0/nayobi/All_datasets/PSI-AVA/TAPIR_trained_models/INSTRUMENTS/checkpoint_best_tools_fold"$FOLD".pyth"  
LANG_CHECK="/media/SSD0/nayobi/Endovis/MICCAI2023/PNG/model_final.pth"
EXP_NAME="TAPIR_"$TEXT_ENCODER

#-------------------------
DATA_VER="psi-ava"
EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
CONFIG_PATH="configs/MVIT_"$TASK".yaml"
MAP_FILE="surgery_"$TASK"_list.pbtxt"
FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

TYPE="pytorch"
#-------------------------
# Run experiment

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=3 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
MODEL.TEXT_ENCODER $TEXT_ENCODER \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.LANG_CHECKPOINT $LANG_CHECK \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
AVA.FRAME_DIR $FRAME_DIR \
AVA.FRAME_LIST_DIR $FRAME_LIST \
AVA.ANNOTATION_DIR $ANNOT_DIR \
AVA.LABEL_MAP_FILE $MAP_FILE \
AVA.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FASTER.FEATURES_TRAIN $FF_TRAIN \
FASTER.FEATURES_VAL $FF_VAL \
TRAIN.PRETRAIN 'cross' \
MODEL.MLP True \
OUTPUT_DIR $OUTPUT_DIR

# #############################################################################################

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="ACTION_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/All_datasets/PSI-AVA/TAPIR_trained_models/INSTRUMENTS/checkpoint_best_tools_fold"$FOLD".pyth"  
# LANG_CHECK="/media/SSD0/nayobi/Endovis/MICCAI2023/PNG/model_final.pth"
# EXP_NAME="TAPIR_"$TEXT_ENCODER

# #-------------------------
# DATA_VER="psi-ava"
# EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
# CONFIG_PATH="configs/MVIT_"$TASK".yaml"
# MAP_FILE="surgery_"$TASK"_list.pbtxt"
# FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
# OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
# ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
# COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
# FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
# FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# mkdir -p $OUTPUT_DIR

# CUDA_VISIBLE_DEVICES=3 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.LANG_CHECKPOINT $LANG_CHECK \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# TRAIN.PRETRAIN 'cross' \
# OUTPUT_DIR $OUTPUT_DIR 

# #######################################################################################333

# # Experiment setup
# FOLD="2" # Fold of the cross-validation split.
# TEXT_ENCODER="biobert"
# TASK="PERMS_GROUND" # Short term tasks "TOOLS" for the instrument detection task or "ACTIONS" for the atomic action recognition task
# CHECKPOINT="/media/SSD0/nayobi/All_datasets/PSI-AVA/TAPIR_trained_models/INSTRUMENTS/checkpoint_best_tools_fold"$FOLD".pyth"  # Path to the model weights of the pretrained model
# LANG_CHECK="/media/SSD0/nayobi/Endovis/MICCAI2023/PNG/model_final.pth"
# EXP_NAME="TAPIR_"$TEXT_ENCODER"_l2"

# #-------------------------
# DATA_VER="psi-ava"
# EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
# CONFIG_PATH="configs/MVIT_"$TASK".yaml"
# MAP_FILE="surgery_"$TASK"_list.pbtxt"
# FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
# OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
# ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
# COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
# FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
# FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# mkdir -p $OUTPUT_DIR

# CUDA_VISIBLE_DEVICES=3 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.LANG_CHECKPOINT $LANG_CHECK \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# MODEL.L2_NORM True \
# TRAIN.PRETRAIN 'cross' \
# OUTPUT_DIR $OUTPUT_DIR   