DATA_PATH="data"
export CUDA_VISIBLE_DEVICES=0,1


#################################################################################
#                                    HRN                                        #
#################################################################################

# Since HRN is based on weight pretrained by DeTecCtive, We need to first download the weights
# For Deepfake dataset
wget -c https://huggingface.co/heyongxin233/DeTeCtive/resolve/main/Deepfake_best.pth ./ckpt/Deepfake_best.pth

# # deepfake
pretrained_pth="./ckpt/Deepfake_best.pth"
python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 5 --lr 2e-5 --warmup_steps 2000\
    --dataset deepfake \
    --method hrn\
    --classifier_dim 1\
    --only_classifier\
    --resum True\
    --pth_path ${pretrained_pth}\
    --model_name princeton-nlp/unsup-simcse-roberta-base --path ${DATA_PATH}/Deepfake/cross_domains_cross_models\
    --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val\

# For M4 dataset
wget -c https://huggingface.co/heyongxin233/DeTeCtive/resolve/main/M4_multilingual_best.pth -c ./ckpt/M4_multilingual_best.pth 
# # M4-multilingual
pretrained_pth="./ckpt/M4_multilingual_best.pth"
python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 10 --lr 2e-5 --warmup_steps 2000\
    --dataset M4 \
    --method hrn\
    --classifier_dim 1\
    --only_classifier\
    --resum True\
    --pth_path ${pretrained_pth}\
    --model_name princeton-nlp/unsup-simcse-roberta-base --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
    --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test\

# # raid
wget -c https://huggingface.co/Shengkun/ood-detection/resolve/main/model_raid.pth -c ./ckpt/model_raid.pth
pretrained_pth="./ckpt/model_raid.pth"
python train_classifier_hrn.py --device_num 2 --per_gpu_batch_size 64 --total_epoch 3 --lr 2e-5 --warmup_steps 100\
    --dataset raid\
    --method hrn\
    --classifier_dim 1\
    --only_classifier\
    --resum True\
    --pth_path ${pretrained_pth}\
    --model_name princeton-nlp/unsup-simcse-roberta-base  \
    --name raid-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name val \
