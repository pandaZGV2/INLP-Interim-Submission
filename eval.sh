PYTHON=python3
CHECKPOINT_PATH=$1
ARCH=vanilla
GPU=1
BATCH_SIZE=2
CORPUS=gcdc
declare -a arr=("Yahoo" "Clinton" "Enron" "Yelp")
MODEL_NAME=roberta-base
ONLINE_MODE=0
TASK=sentence-ordering

for i in "${arr[@]}"
    do
    $PYTHON main.py --checkpoint_path $CHECKPOINT_PATH --inference --arch $ARCH --gpus $GPU --batch_size $BATCH_SIZE --corpus $CORPUS --sub_corpus $i --model_name $MODEL_NAME --freeze_emb_layer  --task $TASK
    done
