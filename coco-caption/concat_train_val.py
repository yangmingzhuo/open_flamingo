import json

train_data = json.load(open("/data/wyl/coco_data/annotations/captions_train2014.json"))
val_data = json.load(open("/data/wyl/coco_data/annotations/captions_val2014.json"))

train_data['annotations'].extend(val_data['annotations'])
train_data['images'].extend(val_data['images'])

json.dump(train_data, open("./captions_trainval2014.json", "w"))