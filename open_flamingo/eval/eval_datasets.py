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
    
class COCOTrainDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/data/wyl/coco_data/train2014",
        val_image_dir_path="/data/wyl/coco_data/val2014",
        annotations_path="/data/wyl/coco_data/annotations/captions_train2014.json",
        val_annotations_path="/data/wyl/coco_data/annotations/captions_val2014.json",
        train_split_path = '/data/wyl/arctic-captions/splits/coco_train.txt',
        train_split_path_2 = '/data/wyl/arctic-captions/splits/coco_restval.txt',
        WC_captions_path="/data/wyl/open_flamingo/wc_vis_80.json",
        WC_best_gt_path="/data/wyl/open_flamingo/coco-caption/best_gt_WC(80).json",
        IP_captions_path="/data/wyl/open_flamingo/cocoresults_train_baseline_32.json",
        IP_best_gt_path="/data/wyl/open_flamingo/coco-caption/best_gt_IP.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.val_image_dir_path = val_image_dir_path
        # use karpathy split 113287
        self.train_names = []
        with open(train_split_path, 'r', encoding='utf-8') as f:
            self.train_names = [i.strip("\n") for i in f.readlines()]
        with open(train_split_path_2, 'r', encoding='utf-8') as f:
            self.train_names.extend([i.strip("\n") for i in f.readlines()])      
        print(len(self.train_names))

        self.imgs = {}
        for tn in self.train_names:
            image_id = int(tn.split("_")[2].split(".")[0])
            self.imgs[image_id] = {"image_id": image_id,
                                   "captions": [],
                                   "image": os.path.join(self.image_dir_path, tn) if tn.find("train") != -1 else os.path.join(val_image_dir_path, tn)}
            
        # add annotation
        self.annotations = json.load(open(annotations_path, "r"))["annotations"]
        self.annotations.extend(json.load(open(val_annotations_path, "r"))["annotations"])
        for ann in self.annotations:
            if self.imgs.get(ann["image_id"]):
                self.imgs[ann["image_id"]]["captions"].append(ann["caption"])

        # add caption generate from image caption model
        self.WC_captions = json.load(open(WC_captions_path, "r"))
        for wc in self.WC_captions:
            if self.imgs.get(wc['image_id']):
                self.imgs[wc['image_id']].update({"WC_captions": [wc["caption"]]})

        # add iterative caption
        self.IP_captions = json.load(open(IP_captions_path, "r"))
        for ip in self.IP_captions:
            if self.imgs.get(ip['image_id']):
                self.imgs[ip['image_id']].update({"IP_captions": [ip["caption"]]})

        # add gt select form WC IP
        self.WC_best_gts = json.load(open(WC_best_gt_path, "r"))
        for wc_gt in self.WC_best_gts:
            if self.imgs.get(wc_gt['image_id']):
                self.imgs[wc_gt['image_id']].update({"WC_gt_idx": wc_gt["gt_idx"]})
        self.IP_best_gts = json.load(open(IP_best_gt_path, "r"))
        for ip_gt in self.IP_best_gts:
            if self.imgs.get(ip_gt['image_id']):
                self.imgs[ip_gt['image_id']].update({"IP_gt_idx": ip_gt["gt_idx"]})
            
        self.images = list(self.imgs.values())
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.images)

    def id2item(self, idx):
        image = Image.open(self.imgs[idx]["image"])
        return {
            "image": image,
            "captions": self.imgs[idx]["captions"],
            "image_id": self.imgs[idx]["image_id"],
            "WC_captions": self.imgs[idx]["WC_captions"],
            "WC_gt_idx": self.imgs[idx]["WC_gt_idx"],
            "IP_captions": self.imgs[idx]["IP_captions"],
            "IP_gt_idx": self.imgs[idx]["IP_gt_idx"],
        }

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]["image"])
        return {
            "image": image,
            "captions": self.images[idx]["captions"],
            "image_id": self.images[idx]["image_id"],
            "WC_captions": self.images[idx]["WC_captions"],
            "WC_gt_idx": self.images[idx]["WC_gt_idx"],
            "IP_captions": self.images[idx]["IP_captions"],
            "IP_gt_idx": self.images[idx]["IP_gt_idx"],
        }
    
class COCOTestDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/data/wyl/coco_data/val2014",
        annotations_path="/data/wyl/coco_data/annotations/captions_val2014.json",
        test_split_path = '/data/wyl/arctic-captions/splits/coco_test.txt',
        clip_ids_path = "/data/wyl/clip/test_topn",
        clip_caps_path = "/data/wyl/clip/test_img2cap_topn",
        # recurrent_path = "/data/wyl/open_flamingo/output_CLIP_new/cocoresults_CLIP_32.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations_path = annotations_path
        self.images = {}
        # use karpathy split 5000
        self.test_names = []
        with open(test_split_path, 'r', encoding='utf-8') as f:
            self.test_names = [i.strip("\n") for i in f.readlines()] 
        print(len(self.test_names))

        self.imgs = {}
        for tn in self.test_names:
            image_id = int(tn.split("_")[2].split(".")[0])
            self.imgs[image_id] = {"image_id": image_id,
                                   "captions": [],
                                   "image": os.path.join(image_dir_path, tn),
                                    "clip_image_ids": [i[0] for i in json.load(open(os.path.join(clip_ids_path, str(image_id)+"_sim.json"), "r"))],
                                    "clip_caps_imgids": [i['image_id'] for i in json.load(open(os.path.join(clip_caps_path, str(image_id)+"_sim.json"), "r"))],
                                    "clip_caps": [i['caption'] for i in json.load(open(os.path.join(clip_caps_path, str(image_id)+"_sim.json"), "r"))]
                                    }
            
        # add coarse caption from recurrent
        # self.recurrent_captions = json.load(open(recurrent_path, "r"))
        # for rc in self.recurrent_captions:
        #     if self.imgs.get(rc["image_id"]):
        #         self.imgs[rc["image_id"]].update({"RC_caption": rc["caption"]})
        #     else:
        #         raise KeyError
        # add annotation
        self.annotations = json.load(open(annotations_path, "r"))["annotations"]
        for ann in self.annotations:
            if self.imgs.get(ann["image_id"]):
                self.imgs[ann["image_id"]]["captions"].append(ann["caption"])
            
        self.images = list(self.imgs.values())
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]["image"])
        return {
            "image": image,
            "captions": self.images[idx]["captions"],
            "image_id": self.images[idx]["image_id"],
            "clip_image_ids": self.images[idx]["clip_image_ids"],
            "clip_caps_imgids": self.images[idx]["clip_caps_imgids"],
            "clip_caps": self.images[idx]["clip_caps"],
            # "RC_caption": self.images[idx]["RC_caption"],
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
