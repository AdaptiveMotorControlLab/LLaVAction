from operator import attrgetter

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


def llava_inference(video_frames, tokenizer, model, image_processor, max_length,  gt,  num_frames=16):

    model.eval()    
    device = "cuda"    
    video_frames = video_frames[0]
    temporal_stride = 16 // num_frames
    video_frames = video_frames[::temporal_stride]
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    conv_template = "qwen_1_5"

    question = gt['question'][0]
    option = gt['option'][0]

    question = f"{DEFAULT_IMAGE_TOKEN}\n{question}:{option}"   

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]
