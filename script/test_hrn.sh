DATA_PATH="data" #PATH for the data
# Model_PATH="ckpt/M4_multilingual_best.pth" #PATH for the model ckpt
# Model_PATH="ckpt/Deepfake_best.pth" #PATH for the model ckpt
Model_PATH="/home/zc/DeTeCtive/runs/M4-multilingual-roberta-base_v5/M4-multilingual-roberta-base_v0/model_classifier_hrn.pth" #PATH for the model ckpt

export CUDA_VISIBLE_DEVICES=0,1
# raid
# python test_dsvdd.py --device_num 2 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode raid --database_name 'train' \
#                    --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/raid


# deepfake
# python test.py --device_num 2 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode deepfake  \
#                    --ood_type hrn \
#                    --num_models 7 \
#                    --out_dim 768 \
#                    --classifier_dim 1\
#                    --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
#                    --model_path ${Model_PATH}

# # TuringBench
# python test_dsvdd.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode Turing --database_path ${DATA_PATH}/TuringBench/AA --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/TuringBench/AA --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/TuringBench

# # M4-monolingual
# python test_dsvdd.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'monolingual_train' \
#                    --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'monolingual_test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/M4-monolingual

# # M4-multilingual
python test.py --device_num 2 --batch_size 128 --max_K 5  --num_models 5 \
                --model_name princeton-nlp/unsup-simcse-roberta-base \
                --out_dim 768 \
                --ood_type hrn \
                --classifier_dim 1\
                --mode M4 \
                --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
                --model_path ${Model_PATH} 

# # OUTFOX,attack:none,outfox,dipper
# python test_dsvdd.py --device_num 8 --batch_size 128 --max_K 51 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode OUTFOX --attack dipper --database_path ${DATA_PATH}/OUTFOX --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/OUTFOX --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/OUTFOX