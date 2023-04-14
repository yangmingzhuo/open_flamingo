from open_flamingo.eval.eval_datasets import COCOTrainDataset
import os

COCO_IMG_PATH="/data/wyl/coco_data/train2014" # coco dataset
COCO_ANNO_PATH="/data/wyl/coco_data/annotations/captions_train2014.json" # coco dataset

coco_data = COCOTrainDataset(COCO_IMG_PATH, COCO_ANNO_PATH)
print(len(os.listdir(coco_data.image_dir_path)))
print(len(coco_data.annotations))