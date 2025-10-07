DATA_PATH="data" #PATH for the data
export CUDA_VISIBLE_DEVICES=0,1
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}
# # deepfake
echo "============================"
echo "[$(timestamp)] Starting test: DeepFake"
echo "============================"
Model_PATH="ckpt/dsvdd/deepfake/model_classifier_best.pth"  #PATH for the model on deepfake dataset
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake  \
                   --ood_type deepsvdd \
                   --out_dim 768 \
                   --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
                   --model_path ${Model_PATH}

echo "============================"
echo "[$(timestamp)] Starting test: M4-Multilingual"
echo "============================"
Model_PATH="ckpt/dsvdd/M4-multilingual/model_classifier_best.pth" #PATH for the model on M4-multilingual dataset
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode M4 \
                   --ood_type deepsvdd \
                   --out_dim 768 \
                   --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
                   --model_path ${Model_PATH} 

# raid
echo "============================"
echo "[$(timestamp)] Starting test: RAID"
echo "============================"
Model_PATH="ckpt/dsvdd/raid/model_classifier_best.pth"     #PATH for the model on raid dataset
python test.py --device_num 2 --batch_size 512 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode raid \
                   --ood_type deepsvdd\
                   --out_dim 768 \
                   --test_dataset_name test\
                   --model_path ${Model_PATH} 

echo "============================"
echo "[$(timestamp)] All tests completed!"