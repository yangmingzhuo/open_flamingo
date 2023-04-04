import json
import os
annotations = json.load(open("/data/wyl/coco_data/annotations/captions_train2014.json", "r"))["annotations"]

images = {}
for ann in annotations:
    if not images.get(ann["image_id"]):
        images[ann["image_id"]] = {
            "captions": [ann["caption"]],
            "image_id": ann["image_id"],
        }
    else:
        images[ann["image_id"]]["captions"].append(ann["caption"])

output = json.load(open('/data/wyl/open_flamingo/output_baseline/baseline_res_32.json', "r"))

out = {}
for i in output:
    out[i["image_id"]] = images[i["image_id"]]

json.dump(list(out.values()), open('gt.json', 'w'), indent=4)