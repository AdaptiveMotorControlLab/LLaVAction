from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import numpy as np
import copy


def llava_ov_process(video_frames, 
    tokenizer, 
    model, 
    image_processor, 
    mc_data,
    clip_length = 16,
    num_frames = 16,
    temperature = 0,
    is_test = False):

    device = "cuda" 
    video_frames = video_frames[0]
    temporal_stride = clip_length // num_frames
    video_frames = video_frames[::temporal_stride]
    image_tensors = []

    # the tensor type needs to be matched   
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
    image_tensors.append(frames)

    conv_template = "qwen_1_5"

    question = mc_data['question'][0]
    options = mc_data['options'][0]

    question = f"{DEFAULT_IMAGE_TOKEN}\n{question}:{options}"     
    
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
        temperature=temperature,
        max_new_tokens=4096,
        modalities=["video"],
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]


def llava_video_process(
    video_frames, 
    tokenizer, 
    model, 
    image_processor, 
    mc_data,
    clip_length = 16,
    num_frames = 16,
    temperature = 0,
    time_meta = {},
    is_test = False):

    device = "cuda"

    video_frames = video_frames[0]

    temporal_stride = clip_length // num_frames

    video_frames = video_frames[::temporal_stride]

    image_tensors = []

    video_duration = time_meta['duration'].item()
    n_frames = time_meta['n_frames'].item()
    frame_time = time_meta['frame_time']
    frame_time = [e[0] for e in frame_time]

    print ("what is meta")
    print ('n_frame', n_frames)
    print ('true video frames', len(video_frames))
    print ('frame_time', frame_time)
    print ('video_duration', video_duration)
    print ('is_test', is_test)

    time_instruciton = f"The video lasts for {video_duration:.2f} seconds, and {n_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."    
    
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)

    image_tensors.append(frames)

    conv_template = "qwen_1_5"

    question = mc_data['question'][0]
    options = mc_data['options'][0]
    
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n:{options}"


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
        temperature=temperature,
        max_new_tokens=4096,
        modalities=["video"],
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]



def llava_inference(
    pretrained_name, 
    video_frames, 
    tokenizer, 
    model, 
    image_processor, 
    mc_data,
    clip_length = 16,
    num_frames = 16,
    temperature = 0,
    is_test = False,
    time_meta = None
    ):

    model.eval()    
       

    if 'ov' in pretrained_name:
        return llava_ov_process(video_frames,
                         tokenizer,
                         model,
                         image_processor,
                         mc_data,
                         clip_length,
                         num_frames,
                         temperature,
                         is_test)
    elif 'Video' in pretrained_name:
        return llava_video_process(
            video_frames, 
            tokenizer, 
            model, 
            image_processor, 
            mc_data,
            clip_length = clip_length,
            num_frames = num_frames,
            temperature = temperature,
            time_meta = time_meta,
            is_test = is_test            
        )
