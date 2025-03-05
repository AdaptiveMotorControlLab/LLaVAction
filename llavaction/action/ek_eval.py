import random
import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
import sys
import os
import numpy as np
from llavaction.action.llava_inference import llava_inference
import json
import logging
from llavaction.utils import rank0_print
from llavaction.action.utils import generate_label_map
from collections import Counter 
import torch.distributed as dist
from llavaction.action.dataset import VideoMultiChoiceDataset
import torchvision.io as io
import re

def process_raw_pred(raw_pred):
    matches = re.findall(r"[A-Z]\.\s(.+)", raw_pred)
    
    if 'None' in raw_pred:
        return raw_pred.replace('None. ', '')
    
    if matches:
        # Get the last match
        last_match = matches[-1]
        # Remove a trailing period and anything after it
        last_match = re.sub(r"\.\s*.*$", "", last_match)
        return last_match
    else:
        return raw_pred

def safe_all_reduce(tensor):
    """Safely perform all_reduce operation with error handling"""
    try:
        # Ensure tensor is on the correct device
        current_device = torch.cuda.current_device()
        if tensor.device != torch.device(f'cuda:{current_device}'):
            tensor = tensor.to(f'cuda:{current_device}')
        
        # Make sure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Create a new process group specifically for this operation
        pg = dist.new_group(ranks=list(range(dist.get_world_size())))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
        dist.destroy_process_group(pg)
        
    except Exception as e:
        print(f"Error in all_reduce: {str(e)}")
        # If all_reduce fails, at least return the local tensor
        return tensor
    
    return tensor


def setup():
    if not dist.is_initialized():
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        # os.environ['MASTER_ADDR'] = '127.0.0.1'
        # os.environ['MASTER_PORT'] = '29500'
        
        # Initialize the process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
        print(f"Process group initialized for rank {rank}")
        
        print ('check master addr', os.environ['MASTER_ADDR'])
        print ('check master port', os.environ['MASTER_PORT'])
               
        
        # Set the local GPU based on the rank
        #local_rank = rank % torch.cuda.device_count()
        #local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print(f"Using GPU {local_rank} for rank {rank}")
        device = torch.device(f"cuda:{local_rank}")
        return device
    
    # Return the appropriate device
    rank = int(os.environ['RANK'])
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    return device

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 cls', add_help=False)
    parser.add_argument('--dataset', default='ek100_cls', type=str, choices=['ek100_mir', 'ek100_cls', 'ego4d', 'ekframes_cls'])
    parser.add_argument('--root', default='/data/EK100/EK100_320p_15sec_30fps_libx264', type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        default='/data/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv')
    parser.add_argument('--val-metadata', type=str,
                        default='/data/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for testing')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for testing')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--norm-style', default='openai', type=str, choices=['openai', 'timm'])
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    
    # llava related
    parser.add_argument('--pretrained_name', default = '', type = str, help ='the name in huggingface')
    parser.add_argument('--llava_num_frames', default=16, type=int, help='number of frames for llava')
    ## avion refinement 
    parser.add_argument('--action_predictions', default=None, type=str, help='path to action predictions')
    parser.add_argument('--topk_predictions', default = 5, type =int)
    parser.add_argument('--llava_checkpoint', default = None, type = str)
    parser.add_argument('--action_representation', default = 'GT_random_narration_cut', type = str, 
                        choices = ['first_sample', 'official_key', 
                                   'random_narration_cut', 'top1_narration_cut', 'topk_narration_cut_key',
                                   'GT_key', 'GT_random_narration', 'GT_random_narration_cut', 'gpt_narration'])
    parser.add_argument('--n_narrations', default = -1, type = int)
    parser.add_argument('--test_type', default = 'base', type = str, choices = ['caption', 'base', 'temporal_cot_caption', 'temporal_cot_pseudo', 'temporal_cot_oracle', 'caption_then_answer', 'direct_narration'])
    parser.add_argument('--learn_neighbor_actions', type= str, default = "")
    parser.add_argument('--pseudo_folder', default = None, type = str)
    parser.add_argument('--output_dir', default = None, type = str)
    parser.add_argument("--perspective", default = "first_person", type = str)
    parser.add_argument('--benchmark_testing', action='store_true', default = False)
    parser.add_argument('--include_time_instruction', action='store_true', default = False)
    parser.add_argument('--gen_type', type = str, default = 'action_model') # action_model, random
    return parser

def prepare_llava(pretrained, use_flash_attention = True):

    import warnings
    warnings.filterwarnings("ignore")
    from llavaction.model.builder import load_pretrained_model    
    model_name = "llava_qwen"

    device_map = "auto"

    overwrite_config = {}
    if 'ov' not in pretrained:
        if 'video' in pretrained or 'Video' in pretrained or '7b' in pretrained:
            overwrite_config =  {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}

    else:
        pass

    if not use_flash_attention:
        overwrite_config['attn_implementation'] = 'sqpa'
    print ('overwrite_config', overwrite_config)
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, 
                                                                        None, 
                                                                        model_name, 
                                                                        torch_dtype="bfloat16", 
                                                                        device_map=device_map, 
                                                                        overwrite_config = overwrite_config)  # Add any other thing you want to pass in llava_model_args


    return tokenizer, model, image_processor, max_length



def ensemble_llava_evaluation(                              
                              gt_name,
                              frames, 
                              tokenizer, 
                              model, 
                              image_processor, 
                              mc_data,
                              clip_length,  
                              num_frames,
                              test_type = 'base',
                              learn_neighbor_actions = "",                             
                              time_meta = None,
                              meta_data = None,
                              perspective = "first_person",
                              include_time_instruction = False
                              ):
    """
    This function tests how consistent the model is if we shuffle the position of the answers
    It also should use a higher temperature so we might get better performance by ensemble
    """
    temperature = 0
    ensemble_k = 1

    if test_type == 'base':
        temperature = 0,
        ensemble_k = 1
               
    # shuffle the options
    options = mc_data['options'][0]
    letters = mc_data['valid_letters']
    avion_pred = mc_data.get('avion_pred', None)
    # each option was in the format of {letter}. {answer}
    preds = []
    for _ in range(ensemble_k):
        # let's just shuffle the options
        random.shuffle(options)
        for idx, (option, letter) in enumerate(zip(options, letters)):
            sep = option.index('.')
            options[idx] = f'{letter}.{option[sep+1:]}'       

        pred = llava_inference(
                            frames, 
                            tokenizer, 
                            model, 
                            image_processor,  
                            mc_data,  
                            test_type = test_type,
                            clip_length = clip_length, 
                            num_frames=num_frames, 
                            temperature = temperature,
                            time_meta = time_meta,
                            learn_neighbor_actions = learn_neighbor_actions,
                            meta_data = meta_data,
                            perspective = perspective,
                            include_time_instruction = include_time_instruction
                            )
        # remove the trailing comma if there is one
        pred = pred.rstrip(',')
        rank0_print('raw output', pred)
        pred = process_raw_pred(pred)
        rank0_print ('llava pred', pred, 'avion_pred', avion_pred, 'gt_name', gt_name) 
        preds.append(pred)
        
    counter = Counter(preds)
    rank0_print ('inspecting the counter', counter)
    rank0_print ('most common', counter.most_common(1)[0][0])

    if counter.most_common(1)[0][0] != gt_name and counter.most_common(1)[0][0][-1] == ",":
        print ('wrong prediction')
        print ('pred', counter.most_common(1)[0][0])
        print ('gt', gt_name)
        print ('---')

    return counter.most_common(1)[0][0] == gt_name, counter.most_common(1)[0][0]



def evaluate_on_EK100(eval_args, 
                      model= None, 
                      tokenizer= None, 
                      image_processor= None,
                      eval_result_folder = None
                      ):

    device = setup()


    if model is not None:
        image_processor = model.get_vision_tower().image_processor

    gpu_val_transform_ls = []

    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    crop_size = 336
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(eval_args.val_metadata).parent,                                                                                            
                                                                                            eval_args.action_representation)
                                                                                            

    if eval_args.action_predictions:
        with open(eval_args.action_predictions, 'r') as f:
            predictions = json.load(f) 
    
    val_dataset = VideoMultiChoiceDataset(
                eval_args.dataset, eval_args.root, eval_args.val_metadata, val_transform_gpu,
                is_training=False, label_mapping=mapping_vn2act,
                num_clips=eval_args.num_clips,
                chunk_len=eval_args.video_chunk_length,
                clip_length=eval_args.clip_length, clip_stride=eval_args.clip_stride,
                threads=eval_args.decode_threads,
                fast_rcc=eval_args.fused_decode_crop, rcc_params=(crop_size, ),
                is_trimmed=not eval_args.dataset == 'charades_ego',
                labels = labels,
                eval_args = eval_args,
                topk_predictions = eval_args.topk_predictions,
                verb_maps = verb_maps,
                noun_maps = noun_maps,
                eval_result_folder = eval_result_folder,
                action_representation = eval_args.action_representation,
                mapping_vn2narration = mapping_vn2narration,
                avion_predictions = predictions if eval_args.action_predictions else None,
                n_narrations = eval_args.n_narrations,
                gen_type = eval_args.gen_type
            )

    def worker_init_fn(worker_id):
        # Calculate a seed unique to each worker
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def collate_fn(batch):
        frames = [item[0] for item in batch]
        mc_data = [item[1] for item in batch]
        time_meta = [item[2] for item in batch]
        global_index = [item[3] for item in batch]

        frames =  np.stack(frames)        

        return frames, mc_data, time_meta, global_index

    if dist.is_initialized():        
        sampler = DistributedSampler(val_dataset,                                      
                                     shuffle=False)
    else:
        sampler = None

    # use custom collate function to avoid default behavior of converting my list of string to list of tuples of strings
    val_dataloader = DataLoader(val_dataset, 
                                collate_fn=collate_fn,
                                sampler = sampler, 
                                batch_size=1, 
                                pin_memory = False,
                                worker_init_fn=worker_init_fn,
                                shuffle=False)    
        
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',  filemode='w')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Set the same format for console handler as well
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    
    logger = logging.getLogger(__name__)

    pretrained = f"{eval_args.pretrained_name}".strip()
    print ('pretrained', pretrained)

    # so we know it's evaluation during training
    finish_early = False #model is not None

    if model is None:
        if args.llava_checkpoint is not None:
            pretrained = eval_args.llava_checkpoint
        tokenizer, model, image_processor, _ = prepare_llava(pretrained)   
       
    global_avion_correct = torch.tensor(0.0, device=device)
    global_running_corrects = torch.tensor(0.0, device=device)
    global_total_samples = torch.tensor(0.0, device=device)

    if eval_args.test_type == 'debug':
        os.makedirs('debug_and_vis', exist_ok = True)


    lookup_table = None
    meta_data = None
    if eval_args.learn_neighbor_actions:
        from llavaction.action.generate_interval_pred import  get_lookup_dict
        if eval_args.test_type.startswith('temporal_cot'):
            lookup_table = get_lookup_dict(eval_args.val_metadata, 
                                           eval_args.action_representation,
                                           test_type = eval_args.test_type, 
                                           pseudo_folder = eval_args.pseudo_folder)


    for idx, (frames, mc_data, time_meta, global_index) in tqdm(enumerate(val_dataloader)):                  
        with torch.no_grad():

            global_index = global_index[0]
            mc_data = mc_data[0]
            time_meta = time_meta[0]
            
            if eval_args.learn_neighbor_actions and lookup_table:
                _id = time_meta['vid_path']
                _id = _id.replace('/', '-')
                uid = f"{_id}_{time_meta['start_second']}_{time_meta['end_second']}"
                meta_data = lookup_table.get(uid, None)
          
            gt_name = mc_data['gt_answer_name'][0]
            local_avion_correct = torch.tensor(0.0, device=device)
            local_running_corrects = torch.tensor(0.0, device=device)
            local_total_samples = torch.tensor(0.0, device=device)            
                
            if eval_args.action_predictions and eval_args.gen_type == 'action_model':
                avion_pred = mc_data['avion_pred']
                if gt_name == avion_pred:               
                    local_avion_correct.add_(1)
                    global_avion_correct.add_(1)
                
            if eval_args.test_type == 'debug':
                squeezed_frames = torch.squeeze(frames)
                io.write_video(os.path.join('debug_and_vis', f"{global_index}.mp4"), squeezed_frames, fps=16)                  

            # we don't want to evaluate the whole thing
            # let's evaluate 1000 samples to get the complete picture       
            if finish_early and idx> (1 / dist.get_world_size()):
                break                     
        
                    
            llava_correct, llava_pred = ensemble_llava_evaluation(
                                                        gt_name,
                                                        frames, 
                                                        tokenizer,
                                                        model,
                                                        image_processor,
                                                        mc_data,
                                                        eval_args.clip_length,
                                                        eval_args.llava_num_frames,
                                                        test_type = eval_args.test_type,  
                                                        learn_neighbor_actions = eval_args.learn_neighbor_actions,                                                    
                                                        time_meta = time_meta,
                                                        meta_data = meta_data,
                                                        perspective = eval_args.perspective,
                                                        include_time_instruction = eval_args.include_time_instruction
                                                        )
                                                        
                                                        


            if eval_args.test_type == 'debug':
                temp = {'gt': gt_name,
                        'llava_pred': llava_pred}

                with open(os.path.join('debug_and_vis', f"{global_index}.json"), 'w') as f:
                    json.dump(temp, f)


            # log the predictions into prediciton analysis
        
            val_dataset.prediction_analysis.log(global_index,
                                                llava_pred,
                                                gt_name,
                                                mc_data.get('all_avion_preds', None),
                                                time_meta['start_second'],
                                                time_meta['end_second'],
                                                time_meta['vid_path'],
                                                dataset_name = 'EK100')

        
            local_running_corrects.add_(llava_correct)
            global_running_corrects.add_(llava_correct)
                                                                
            local_total_samples.add_(1)
            global_total_samples.add_(1)

            torch.cuda.empty_cache()
            # logger.info(f'Process {dist.get_rank()} - local_total_samples: {local_total_samples:.4f}')
            # logger.info(f'Process {dist.get_rank()} - loca_llava_correct: {llava_correct:.4f}')
            # logger.info(f'Process {dist.get_rank()} - local_avion_corrects: {local_avion_correct:.4f}')
            # logger.info(f'Process {dist.get_rank()} - local_running_corrects: {local_running_corrects:.4f}')
            

    dist.all_reduce(global_running_corrects, op=dist.ReduceOp.SUM)
    #global_running_corrects = safe_all_reduce(global_running_corrects)
    dist.all_reduce(global_total_samples, op=dist.ReduceOp.SUM)
    #global_total_samples = safe_all_reduce(global_total_samples)
    if eval_args.action_predictions:
        dist.all_reduce(global_avion_correct, op=dist.ReduceOp.SUM)
        #global_avion_correct = safe_all_reduce(global_avion_correct)

    # Calculate global accuracy after reduction
    global_accuracy = global_running_corrects.item() / global_total_samples.item()
    if eval_args.action_predictions:
        global_avion_accuracy = global_avion_correct.item() / global_total_samples.item()

    # Ensure only the main process (rank 0) prints the final result
    if dist.get_rank() == 0:
        if eval_args.action_predictions:
            logger.info(f'Global Avion Accuracy: {global_avion_accuracy:.4f}')
        logger.info(f'Final Global Accuracy: {global_accuracy:.4f}')

    val_dataset.prediction_analysis.save()
    
    return global_accuracy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    output_dir = args.output_dir
    evaluate_on_EK100(args, eval_result_folder=output_dir if output_dir is not None else None)
   
