from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import numpy as np
import copy
from llava.action.utils import format_llava_prompt
from llava.utils import rank0_print


def llava_inference(
    video_frames, 
    tokenizer, 
    model, 
    image_processor, 
    mc_data,
    clip_length = 16,
    num_frames = 16,
    temperature = 0,
    test_type = 'base',
    time_meta = None,
    learn_neighbor_actions = "",
    meta_data = None,
    perspective = "first_person"
    ):

        model.eval()              
        device = "cuda"
        # this [0] is only for batch size 1.
        video_frames = video_frames[0]

        temporal_stride = clip_length // num_frames

        video_frames = video_frames[::temporal_stride]

        image_tensors = []

        video_duration = time_meta['duration']
        n_frames = time_meta['n_frames']
        
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)

        image_tensors.append(frames)

        conv_template = "qwen_1_5"

        options = mc_data['options'][0]
        
        if test_type == 'base':
            question_type = "mc_top5_official_key"
        elif test_type == "direct_narration":
            question_type = "direct_narration"
        elif test_type == 'caption' or test_type == 'debug':
            question_type = "caption"
        elif test_type == 'temporal_cot_pseudo':
            question_type = 'temporal_cot_pseudo'
        elif test_type == 'temporal_cot_oracle':
            question_type = 'temporal_cot_oracle'            
        elif test_type == 'temporal_cot_caption':
            question_type = 'temporal_cot_caption'
                    
        if  test_type == 'caption_then_answer':        
            caption_answer = llava_inference([video_frames], 
            tokenizer, 
            model,  
            image_processor, 
            mc_data,
            test_type = 'caption',
            clip_length = clip_length,
            num_frames = num_frames,
            temperature = 0,
            time_meta = time_meta)

            question = format_llava_prompt(DEFAULT_IMAGE_TOKEN,
                                        options,
                                        video_duration,                                  
                                        n_frames,
                                        "mc_top5_official_key",
                                        include_frame_time = False,
                                        learn_neighbor_actions = learn_neighbor_actions,
                                        perspective = perspective,
                                        include_time_instruction= False)

            question = f"You observed the video before and wrote down the notes: {caption_answer}. Now you watch the same video again and you can do better. " +  question                             
            
        else:                        
            question = format_llava_prompt(DEFAULT_IMAGE_TOKEN,
                                        options,
                                        video_duration,                                  
                                        n_frames,
                                        question_type,
                                        include_frame_time = False,
                                        learn_neighbor_actions = learn_neighbor_actions,
                                        include_time_instruction= False,
                                        perspective = perspective,
                                        meta_data=meta_data)


        #rank0_print ("debugging", question)

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