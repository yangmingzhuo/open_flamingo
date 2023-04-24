from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annFile = '/data/wyl/open_flamingo/coco-caption/captions_trainval2014.json'
resFile = '/data/wyl/open_flamingo/cocoresults_train_baseline_0.json'
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

coco_eval = COCOEvalCap(coco, cocoRes)
coco_eval.params["image_id"] = cocoRes.getImgIds()
coco_eval.evaluate()