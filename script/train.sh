
# # deepfake
# python train_classifier.py --device_num 8 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
#     --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test

# # TuringBench
# python train_classifier.py --device_num 8 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset TuringBench --path ${DATA_PATH}/TuringBench/AA \
#     --name TuringBench-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test

# # M4-monolingual
# python train_classifier.py --device_num 8 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-monolingual-roberta-base --freeze_embedding_layer --database_name monolingual_train --test_dataset_name monolingual_test

# # M4-multilingual
# python train_classifier.py --device_num 8 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test

# # OUTFOX
# python train_classifier.py --device_num 8 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset OUTFOX --path ${DATA_PATH}/OUTFOX \
#     --name OUTFOX-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test

DATA_PATH="data"
export CUDA_VISIBLE_DEVICES=0

# DeepSVDD
# deepfake
# python train_classifier.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --out_dim 768\
#     --objective one-class\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
#     --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test

# M4-monolingual
# python train_classifier.py --device_num 2 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --out_dim 768\
#     --objective one-class\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-monolingual-roberta-base --freeze_embedding_layer --database_name monolingual_train --test_dataset_name monolingual_test

# HRN
# deepfake
python train_classifier_hrn.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 5 --lr 2e-5 --warmup_steps 2000\
    --classifier_dim 1\
    --one_loss\
    --only_classifier\
    --resum True\
    --pth_path /home/zc/DeTeCtive/ckpt/Deepfake_best.pth\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
    --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test
