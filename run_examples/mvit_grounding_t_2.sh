# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_3negatives/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_3negatives"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# ###################################################################################3333

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_5negatives/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_5negatives"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# ###################################################################################3333

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_pretrain_3negatives/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_pretrain_3negatives"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# ###################################################################################3333

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_full_pretrain_3negatives/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_full_pretrain_3negatives"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# ###################################################################################3333

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_actions/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_actions"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# ##################################################################################################

# # Experiment setup
# FOLD="2" 
# TEXT_ENCODER="biobert"
# TASK="GROUND_INFERE"
# PRETRAIN_TASK="PERMS_GROUND"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_pretrain/Fold2/checkpoint_best_mean.pyth"
# EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_pretrain"

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

# CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# MODEL.TEXT_ENCODER $TEXT_ENCODER \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE True \
# TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# AVA.FRAME_DIR $FRAME_DIR \
# AVA.FRAME_LIST_DIR $FRAME_LIST \
# AVA.ANNOTATION_DIR $ANNOT_DIR \
# AVA.LABEL_MAP_FILE $MAP_FILE \
# AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FASTER.FEATURES_TRAIN $FF_TRAIN \
# FASTER.FEATURES_VAL $FF_VAL \
# OUTPUT_DIR $OUTPUT_DIR 

# # ##################################################################################################

# # # Experiment setup
# # FOLD="2" 
# # TEXT_ENCODER="biobert"
# # TASK="GROUND_INFERE"
# # PRETRAIN_TASK="PERMS_GROUND"
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_3negatives/Fold2/checkpoint_best_mean.pyth"
# # EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_3negatives"

# # #-------------------------
# # DATA_VER="psi-ava"
# # EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
# # CONFIG_PATH="configs/MVIT_"$TASK".yaml"
# # MAP_FILE="surgery_"$TASK"_list.pbtxt"
# # FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
# # OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
# # FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
# # ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
# # COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
# # FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
# # FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

# # TYPE="pytorch"
# # #-------------------------
# # # Run experiment

# # mkdir -p $OUTPUT_DIR

# # CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# # --cfg $CONFIG_PATH \
# # NUM_GPUS 1 \
# # MODEL.TEXT_ENCODER $TEXT_ENCODER \
# # TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# # TRAIN.CHECKPOINT_EPOCH_RESET True \
# # TRAIN.CHECKPOINT_TYPE $TYPE \
# # TEST.ENABLE False \
# # TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# # AVA.FRAME_DIR $FRAME_DIR \
# # AVA.FRAME_LIST_DIR $FRAME_LIST \
# # AVA.ANNOTATION_DIR $ANNOT_DIR \
# # AVA.LABEL_MAP_FILE $MAP_FILE \
# # AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# # BN.NUM_BATCHES_PRECISE 72 \
# # FASTER.FEATURES_TRAIN $FF_TRAIN \
# # FASTER.FEATURES_VAL $FF_VAL \
# # OUTPUT_DIR $OUTPUT_DIR 

# # ##################################################################################################

# # # Experiment setup
# # FOLD="2" 
# # TEXT_ENCODER="biobert"
# # TASK="GROUND_INFERE"
# # PRETRAIN_TASK="PERMS_GROUND"
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_3negatives/Fold2/checkpoint_best_mean.pyth"
# # EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_3negatives_loss_per_task"

# # #-------------------------
# # DATA_VER="psi-ava"
# # EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
# # CONFIG_PATH="configs/MVIT_"$TASK".yaml"
# # MAP_FILE="surgery_"$TASK"_list.pbtxt"
# # FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
# # OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
# # FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
# # ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
# # COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
# # FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
# # FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

# # TYPE="pytorch"
# # #-------------------------
# # # Run experiment

# # mkdir -p $OUTPUT_DIR

# # CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# # --cfg $CONFIG_PATH \
# # NUM_GPUS 1 \
# # MODEL.TEXT_ENCODER $TEXT_ENCODER \
# # TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# # TRAIN.CHECKPOINT_EPOCH_RESET True \
# # TRAIN.CHECKPOINT_TYPE $TYPE \
# # TEST.ENABLE False \
# # TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# # AVA.FRAME_DIR $FRAME_DIR \
# # AVA.FRAME_LIST_DIR $FRAME_LIST \
# # AVA.ANNOTATION_DIR $ANNOT_DIR \
# # AVA.LABEL_MAP_FILE $MAP_FILE \
# # AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# # BN.NUM_BATCHES_PRECISE 72 \
# # FASTER.FEATURES_TRAIN $FF_TRAIN \
# # FASTER.FEATURES_VAL $FF_VAL \
# # MODEL.GROUND_LOSS_PER_TASK True \
# # OUTPUT_DIR $OUTPUT_DIR 

# # ##################################################################################################

# # # Experiment setup
# # FOLD="2" 
# # TEXT_ENCODER="biobert"
# # TASK="GROUND_INFERE"
# # PRETRAIN_TASK="PERMS_GROUND"
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_3negatives/Fold2/checkpoint_best_mean.pyth"
# # EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_3negatives_per_task_layers"

# # #-------------------------
# # DATA_VER="psi-ava"
# # EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
# # CONFIG_PATH="configs/MVIT_"$TASK".yaml"
# # MAP_FILE="surgery_"$TASK"_list.pbtxt"
# # FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
# # OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
# # FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
# # ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
# # COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
# # FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
# # FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

# # TYPE="pytorch"
# # #-------------------------
# # # Run experiment

# # mkdir -p $OUTPUT_DIR

# # CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
# # --cfg $CONFIG_PATH \
# # NUM_GPUS 1 \
# # MODEL.TEXT_ENCODER $TEXT_ENCODER \
# # TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# # TRAIN.CHECKPOINT_EPOCH_RESET True \
# # TRAIN.CHECKPOINT_TYPE $TYPE \
# # TEST.ENABLE False \
# # TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
# # AVA.FRAME_DIR $FRAME_DIR \
# # AVA.FRAME_LIST_DIR $FRAME_LIST \
# # AVA.ANNOTATION_DIR $ANNOT_DIR \
# # AVA.LABEL_MAP_FILE $MAP_FILE \
# # AVA.COCO_ANN_DIR $COCO_ANN_PATH \
# # BN.NUM_BATCHES_PRECISE 72 \
# # FASTER.FEATURES_TRAIN $FF_TRAIN \
# # FASTER.FEATURES_VAL $FF_VAL \
# # MODEL.GROUND_LAYERS_PER_TASK True \
# # OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################

# Experiment setup
FOLD="2" 
TEXT_ENCODER="biobert"
TASK="GROUND_INFERE"
PRETRAIN_TASK="PERMS_GROUND"
CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/TAPIR/outputs/log/"$PRETRAIN_TASK"/TAPIR_"$TEXT_ENCODER"_3negatives/Fold2/checkpoint_best_mean.pyth"
EXP_NAME="TAPIR_"$PRETRAIN_TASK"_"$TEXT_ENCODER"_3negatives_layers"

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

CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
MODEL.TEXT_ENCODER $TEXT_ENCODER \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.PRETRAIN_TASK $PRETRAIN_TASK \
AVA.FRAME_DIR $FRAME_DIR \
AVA.FRAME_LIST_DIR $FRAME_LIST \
AVA.ANNOTATION_DIR $ANNOT_DIR \
AVA.LABEL_MAP_FILE $MAP_FILE \
AVA.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FASTER.FEATURES_TRAIN $FF_TRAIN \
FASTER.FEATURES_VAL $FF_VAL \
MODEL.GROUND_LAYERS_TASK_LAYERS True \
OUTPUT_DIR $OUTPUT_DIR 