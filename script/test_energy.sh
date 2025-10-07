DATA_PATH="data" #PATH for the data
export CUDA_VISIBLE_DEVICES=0,1

# deepfake
Model_PATH="ckpt/energy/deepfake/model_classifier_energy_best.pth" #PATH for the model ckpt
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake  \
                   --ood_type energy \
                   --out_dim 768 \
                   --classifier_dim 7\
                   --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
                   --model_path ${Model_PATH}

# # # M4-multilingual
Model_PATH=""
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode M4 \
                   --ood_type energy \
                   --out_dim 768 \
                   --classifier_dim 5\
                   --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'multilingual_train' \
                   --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
                   --model_path ${Model_PATH} 

# # # raid
# Model_PATH=""
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode raid \
                   --ood_type energy \
                   --out_dim 768 \
                   --classifier_dim 6\
                   --database_name 'train' \
                   --test_dataset_name 'test'\
                   --model_path ${Model_PATH} 