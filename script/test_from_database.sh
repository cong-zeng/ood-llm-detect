DATA_PATH="/opt/LLM_detect_data" #PATH for the data
Model_PATH="/opt/DeTeCtive/pth/Deepfake_best.pth" #PATH for the model ckpt
DATABASE_PATH="/opt/DeTeCtive/database/deepfake" #PATH for the database

# deepfake
python test_from_database.py --device_num 8 --batch_size 64 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake --database_path ${DATABASE_PATH} \
                   --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
                   --model_path ${Model_PATH}

# # TuringBench
# python test_from_database.py --device_num 8 --batch_size 64 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                      --mode Turing --database_path ${DATABASE_PATH} \
#                      --test_dataset_path ${DATA_PATH}/TuringBench/AA --test_dataset_name 'test'\
#                      --model_path ${Model_PATH}

# # M4-monolingual
# python test_from_database.py --device_num 8 --batch_size 64 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                      --mode M4 --database_path ${DATABASE_PATH} \
#                      --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'monolingual_test'\
#                      --model_path ${Model_PATH}

# # M4-multilingual
# python test_from_database.py --device_num 8 --batch_size 64 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                      --mode M4 --database_path ${DATABASE_PATH} \
#                      --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
#                      --model_path ${Model_PATH}

# # OUTFOX,attack:none,outfox,dipper
# python test_from_database.py --device_num 8 --batch_size 64 --max_K 51 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                      --mode OUTFOX --attack dipper --database_path ${DATABASE_PATH} \
#                      --test_dataset_path ${DATA_PATH}/OUTFOX --test_dataset_name 'test'\
#                      --model_path ${Model_PATH}