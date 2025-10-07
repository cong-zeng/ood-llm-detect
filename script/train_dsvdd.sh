DATA_PATH="data"
export CUDA_VISIBLE_DEVICES=0,1

#################################################################################
#                                    DeepSVDD                                   #
#################################################################################
# deepfake
python train_classifier_dsvdd.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
    --method dsvdd\
    --out_dim 768\
    --objective one-class\
    --one_loss\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
    --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val

# M4-multilingual
# python train_classifier_dsvdd.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --method dsvdd\
#     --out_dim 768\
#     --one_loss\
#     --objective one-class\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test

# raid
# python train_classifier_dsvdd.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 1000\
#     --method dsvdd\
#     --out_dim 768\
#     --one_loss\
#     --objective one-class\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset raid \
#     --name raid-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val
