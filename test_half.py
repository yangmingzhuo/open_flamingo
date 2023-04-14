from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import time

llama_path = "/data/share/pyz/llm_deploy/checkpoints/llama-7b"

device = 'cpu'

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=llama_path,
    tokenizer_path=llama_path,
    cross_attn_every_n_layers=4,
    # new params
    inference=True,
    precision='fp32',
    device=device,
    checkpoint_path="/data/share/OpenFlamingo/openflamingo_checkpoint.pt",
)

# grab model checkpoint from huggingface hub
import torch

from PIL import Image
import requests

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/train2017/000000568041.jpg", stream=True
    ).raw
)

# demo_image_two = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/val2014/000000391895.jpg",
#         stream=True
#     ).raw
# )

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/train2017/000000391895.jpg", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
time_begin = time.time()
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>Output:A man riding a motorcycle on a dirt road.<|endofchunk|><image>Output:"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x.to(device),
    lang_x=lang_x["input_ids"].to(device),
    attention_mask=lang_x["attention_mask"].to(device),
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]), "time: ", time.time() - time_begin)