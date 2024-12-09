import os
import sys
sys.path[0] = os.path.dirname(sys.path[0])

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

# from PIL import Image
# import requests
# import copy
# import torch

# import sys
# import warnings



# warnings.filterwarnings("ignore")
# pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
# model_name = "llava_qwen"
# device = "cuda"
# device_map = "auto"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

# model.eval()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# image_tensor = process_images([image], image_processor, model.config)
# image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
# question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [image.size]


# cont = model.generate(
#     input_ids,
#     images=image_tensor,
#     image_sizes=image_sizes,
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=4096,
# )
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs)




from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import transformers
import ast
import re

from llava.train.train import ModelArguments, DataArguments, TrainingArguments, EK100EvalArguments, LazySupervisedDataset

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EK100EvalArguments))
model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()


os.environ["HF_HOME"] = "huggingface"

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# pretrained = "/mnt/SV_storage/VFM/LLaVA-NeXT/experiments/EK100_quick_config"
model_base = None
model_name = "llava_qwen"

# pretrained = "/mnt/SV_storage/VFM/LLaVA-NeXT/experiments/EK100_lora_quick_check"
# model_base = "/mnt/SV_storage/VFM/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-0.5b-ov/snapshots/381d9947148efb1e58a577f451c05705ceec666e"
# model_name = "lora_llava_qwen"
device = "cuda"
device_map = "auto"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, device_map=device_map, attn_implementation="sdpa")
overwrite_config = {}
if model_args.vision_supervision is not None:
    overwrite_config["vision_supervision"] = model_args.vision_supervision
    overwrite_config["action_types"] = model_args.action_types
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, 
                                                                      device_map=device_map, attn_implementation="flash_attention_2", overwrite_config=overwrite_config)
# model.eval()


vision_tower = model.get_vision_tower()
data_args.image_processor = vision_tower.image_processor
data_args.is_multimodal = True
data_args.mm_use_im_start_end = False
if data_args.image_grid_pinpoints is not None:
    if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
        try:
            patch_size = data_args.image_processor.size[0]
        except Exception as e:
            patch_size = data_args.image_processor.size["shortest_edge"]

        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    elif isinstance(data_args.image_grid_pinpoints, str):
        data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args, eval_args = eval_args)

data = train_dataset[0]

input_ids = data["input_ids"].unsqueeze(0).to(device)
labels = data["labels"].unsqueeze(0).to(device)
images = [data["image"][0][0].half().to(device)]
image_sizes = [data["image"][0][1]]
actions = torch.stack([data["image"][0][3].to(device)])
attention_mask = torch.ones_like(input_ids).bool().to(device)
modalities=["video"]

cont = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    images=images,
    image_sizes=image_sizes,
    modalities=modalities,
    labels=labels,
    actions=actions,
)

aa = 2


# # Function to extract frames from video
# def load_video(video_path, max_frames_num):
#     if type(video_path) == str:
#         vr = VideoReader(video_path, ctx=cpu(0))
#     else:
#         vr = VideoReader(video_path[0], ctx=cpu(0))
#     total_frame_num = len(vr)
#     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
#     frame_idx = uniform_sampled_frames.tolist()
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     return spare_frames  # (frames, height, width, channels)


# # Load and process video
# video_path = "docs/jobs.mp4"
# video_frames = load_video(video_path, 16)
# print(video_frames.shape) # (16, 1024, 576, 3)
# image_tensors = []
# frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
# image_tensors.append(frames)

# # Prepare conversation input
# conv_template = "qwen_1_5"
# question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [frame.size for frame in video_frames]

# # Generate response
# cont = model.generate(
#     input_ids,
#     images=image_tensors,
#     image_sizes=image_sizes,
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=4096,
#     modalities=["video"],
# )
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs[0])