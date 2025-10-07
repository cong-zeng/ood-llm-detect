DATA_PATH="data"
export CUDA_VISIBLE_DEVICES=0,1


#################################################################################
#                                    HRN                                        #
#################################################################################
# # deepfake
pretrained_pth=""
python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 5 --lr 2e-5 --warmup_steps 2000\
    --dataset deepfake \
    --method hrn\
    --classifier_dim 1\
    --only_classifier\
    --resum True\
    --pth_path ${pretrained_pth}\
    --model_name princeton-nlp/unsup-simcse-roberta-base --path ${DATA_PATH}/Deepfake/cross_domains_cross_models\
    --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val\

# # M4-multilingual
# pretrained_pth=""
# python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 10 --lr 2e-5 --warmup_steps 2000\
#     --dataset M4 \
#     --method hrn\
#     --classifier_dim 1\
#     --only_classifier\
#     --resum True\
#     --pth_path ${pretrained_pth}\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test\

# # raid
# pretrained_pth=""
# python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 64 --total_epoch 3 --lr 2e-5 --warmup_steps 100\
#     --dataset raid\
#     --method hrn\
#     --classifier_dim 1\
#     --only_classifier\
#     --resum True\
#     --pth_path ${pretrained_pth}\
#     --model_name princeton-nlp/unsup-simcse-roberta-base  \
#     --name raid-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val \
