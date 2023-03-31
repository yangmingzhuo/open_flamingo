LM_PATH="/data/share/pyz/llm_deploy/checkpoints/llama-7b" # llama model path
LM_TOKENIZER_PATH="/data/share/pyz/llm_deploy/checkpoints/llama-7b" # llama model path
CKPT_PATH="/data/share/OpenFlamingo/openflamingo_checkpoint.pt"
# checkpoint model path you can run checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt") to get
DEVICE=5 # gpu num

COCO_IMG_PATH="/data/wyl/coco_data/train2014" # coco dataset
COCO_ANNO_PATH="/data/wyl/coco_data/annotations/captions_train2014.json" # coco dataset

RANDOM_ID=$$
RESULTS_FILE="results_${RANDOM_ID}.json"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --device $DEVICE \
    --coco_image_dir_path $COCO_IMG_PATH \
    --coco_annotations_json_path $COCO_ANNO_PATH \
    --results_file $RESULTS_FILE \
    --num_samples 5000 --shots 16 --num_trials 1 --batch_size 1\
    --cross_attn_every_n_layers 4\
    --eval_coco
    
echo "evaluation complete! results written to ${RESULTS_FILE}"
