import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.imagenet_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class COCOFlickrDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/COCO_train2014_{self.annotations[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"],
        }
    
class COCODataset(Dataset):
    def __init__(
        self,
        image_dir_path="/data/wyl/coco_data/train2014",
        annotations_path="/data/wyl/coco_data/annotations/captions_train2014.json",
        clip_captions_path="/data/wyl/clip/clip_top32.json",
        RT_captions_path="/data/wyl/ImageCaptioning.pytorch/data/coco_ret_caps_cider",
        IC_captions_path="/data/wyl/open_flamingo/vis.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path, "r"))["annotations"]
        self.images = {}
        self.image_ids = []
        self.clip_top32 = json.load(open(clip_captions_path, "r"))
        self.IC_captions = json.load(open(IC_captions_path, "r"))

        for ann in self.annotations:
            if not self.images.get(ann["image_id"]):
                self.images[ann["image_id"]] = {
                    "captions": [ann["caption"]],
                    "image_id": ann["image_id"],
                    "clip_captions": self.clip_top32[str(ann["image_id"])],
                    "RT_captions": json.load(open(os.path.join(RT_captions_path, str(ann["image_id"]) + ".json"), "r"))["ret"],
                }
            else:
                self.images[ann["image_id"]]["captions"].append(ann["caption"])
            self.image_ids.append(ann["image_id"])

        for ic in self.IC_captions:
            if self.images.get(ic['image_id']):
                self.images[ic['image_id']].update({"IC_captions": [ic["caption"]]})
            
        self.images = list(self.images.values())
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.images)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.images[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/COCO_train2014_{self.images[idx]['image_id']:012d}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        img = self.images[idx]
        captions = img["captions"]
        clip_captions = img["clip_captions"]
        RT_captions = img["RT_captions"]
        IC_captions = img["IC_captions"]
        image_id = img["image_id"]
        return {
            "image": image,
            "captions": captions,
            "clip_captions": clip_captions,
            "RT_captions": RT_captions,
            "IC_captions": IC_captions,
            "image_id": image_id,
        }


class VQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
        question_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
        vqa_dataset="vqa",
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.vqa_dataset = vqa_dataset

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.vqa_dataset == "vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_train2014_{question['image_id']:012d}.jpg"
            )
        elif self.vqa_dataset == "ok_vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_val2014_{question['image_id']:012d}.jpg"
            )
        else:
            raise Exception(f"Unknown VQA dataset {self.vqa_dataset}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
