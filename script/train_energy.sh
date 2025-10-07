DATA_PATH="data"
export CUDA_VISIBLE_DEVICES=0,1

#################################################################################
#                                   Energy                                      #
#################################################################################
# deepfake
Model_pth=""
python train_classifier_energy.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 1000\
    --method energy\
    --classifier_dim 7\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
    --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test\
    --resum True\
    --pth_path ${Model_pth}\

# M4-multilingual
Model_pth=""
python train_classifier_energy.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
    --method energy\
    --classifier_dim 5\
    --only_classifier\
    --resum True\
    --pth_path ${Model_pth}\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
    --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test

# raid
Model_pth=""
python train_classifier_energy.py --device_num 2 --per_gpu_batch_size 64 --total_epoch 50 --lr 2e-5 --warmup_steps 1000\
    --method energy\
    --classifier_dim 6\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset raid \
    --name raid-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test\
    --resum True\
    --pth_path ${Model_pth}\