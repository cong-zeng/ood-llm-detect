DATA_PATH="/opt/LLM_detect_data" #PATH for the data
Model_PATH="/opt/DeTeCtive/pth/Deepfake_best.pth" #PATH for the model ckpt

# deepfake
python test_knn.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
                   --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
                   --model_path ${Model_PATH} --save_database --save_path database/deepfake

# # TuringBench
# python test_knn.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode Turing --database_path ${DATA_PATH}/TuringBench/AA --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/TuringBench/AA --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/TuringBench

# # M4-monolingual
# python test_knn.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'monolingual_train' \
#                    --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'monolingual_test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/M4-monolingual

# # M4-multilingual
# python test_knn.py --device_num 8 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'multilingual_train' \
#                    --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/M4-multilingual

# # OUTFOX,attack:none,outfox,dipper
# python test_knn.py --device_num 8 --batch_size 128 --max_K 51 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode OUTFOX --attack dipper --database_path ${DATA_PATH}/OUTFOX --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/OUTFOX --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/OUTFOX