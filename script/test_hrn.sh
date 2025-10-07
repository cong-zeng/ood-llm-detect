DATA_PATH="data" #PATH for the data
export CUDA_VISIBLE_DEVICES=0,1


# deepfake
Model_PATH="" #PATH for the model ckpt
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake  \
                   --ood_type hrn \
                   --num_models 7 \
                   --out_dim 768 \
                   --classifier_dim 1\
                   --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
                   --model_path ${Model_PATH}

# # M4-multilingual
Model_PATH="" #PATH for the model ckpt
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                --mode M4 \
                --ood_type hrn \
                --num_models 5 \
                --out_dim 768 \
                --classifier_dim 1\
                --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
                --model_path ${Model_PATH} 
                
# # raid
Model_PATH="" #PATH for the model ckpt
python test_dsvdd.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode raid \
                   --ood_type hrn \
                   --num_models 6 \
                   --out_dim 768 \
                   --classifier_dim 1\
                   --test_dataset_name 'test'\
                   --model_path ${Model_PATH} 