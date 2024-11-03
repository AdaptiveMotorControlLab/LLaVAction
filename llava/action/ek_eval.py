import random
import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
import sys
import os
from llava.action.llava_inference import llava_inference
import json
import logging
from llava.utils import rank0_print
from llava.action.utils import generate_label_map,  match_answer, remove_option_letter
from collections import Counter 
import torch.distributed as dist
from llava.action.dataset import VideoMultiChoiceDataset


def setup(rank, world_size):
    # Check if the process group is already initialized
    if not dist.is_initialized():
        # Initialize the process group if it hasn't been initialized yet
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # Replace with master node IP
        os.environ['MASTER_PORT'] = '29500'      # Set a port for communication
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"Process group initialized for rank {rank}")
        
        # Set the GPU device based on rank
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print(f"Using GPU {local_rank} for rank {rank}")


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 cls', add_help=False)
    parser.add_argument('--dataset', default='ek100_cls', type=str, choices=['ek100_mir'])
    parser.add_argument('--root', default='/data/EK100/EK100_320p_15sec_30fps_libx264', type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        default='/data/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv')
    parser.add_argument('--val-metadata', type=str,
                        default='/data/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
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
                                   'GT_key', 'GT_random_narration', 'GT_random_narration_cut'])
    parser.add_argument('--n_narrations', default = -1, type = int)
    parser.add_argument('--ensemble_test', action='store_true')

    
    return parser

def prepare_llava(pretrained):

    import warnings
    warnings.filterwarnings("ignore")
    from llava.model.builder import load_pretrained_model    
    model_name = "llava_qwen"

    device_map = "auto"

    overwrite_config = None
    if 'video' in pretrained:
        overwrite_config =  {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}


    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, 
                                                                          None, 
                                                                          model_name, 
                                                                          torch_dtype="bfloat16", 
                                                                          device_map=device_map, 
                                                                          overwrite_config = overwrite_config)  # Add any other thing you want to pass in llava_model_args


    return tokenizer, model, image_processor, max_length



def ensemble_llava_evaluation(
                              pretrained_name,
                              gt_name,
                              frames, 
                              tokenizer, 
                              model, 
                              image_processor, 
                              mc_data,
                              clip_length,  
                              num_frames,
                              temperature = 0,
                              ensemble_k = 1,
                              time_meta = None,
                              ):
    """
    This function tests how consistent the model is if we shuffle the position of the answers
    It also should use a higher temperature so we might get better performance by ensemble
    """

    # shuffle the options
    options = mc_data['options'][0]
    letters = mc_data['valid_letters']
    avion_pred = mc_data['avion_pred']
    # each option was in the format of {letter}. {answer}
    preds = []
    for _ in range(ensemble_k):
            # let's just shuffle the options
        random.shuffle(options)
        for idx, (option, letter) in enumerate(zip(options, letters)):
            sep = option.index('.')
            options[idx] = f'{letter}.{option[sep+1:]}'       

        pred = llava_inference(
                            pretrained_name,
                            frames, 
                            tokenizer, 
                            model, 
                            image_processor,  
                            mc_data,  
                            clip_length = clip_length, 
                            num_frames=num_frames, 
                            temperature = temperature,
                            time_meta = time_meta
                               )
        rank0_print('raw output', pred)
        pred = remove_option_letter(pred)
        rank0_print ('llava pred', pred, 'avion_pred', avion_pred, 'gt_name', gt_name) 
        preds.append(pred)
        
    counter = Counter(preds)
    rank0_print ('inspecting the counter', counter)
    rank0_print ('most common', counter.most_common(1)[0][0])

    return counter.most_common(1)[0][0] == gt_name, counter.most_common(1)[0][0]



def evaluate_on_EK100(eval_args, 
                      model= None, 
                      tokenizer= None, 
                      image_processor= None,
                      eval_result_folder = None
                      ):

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    setup(rank, world_size)


    if model is not None:
        image_processor = model.get_vision_tower().image_processor

    gpu_val_transform_ls = []

    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    crop_size = 336
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(eval_args.val_metadata).parent,                                                                                            
                                                                                            eval_args.action_representation,
                                                                                            cache_file =  Path(eval_args.val_metadata).parent / 'nlp_cache.pkl')

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
            )

    def collate_fn(batch):
        frames = [item[0] for item in batch]
        mc_data = [item[1] for item in batch]
        time_meta = [item[2] for item in batch]
        global_index = [item[3] for item in batch]

        frames =  torch.stack(frames)        

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

    pretrained = f"lmms-lab/{eval_args.pretrained_name}".strip()
    print ('pretrained', pretrained)

    # so we know it's evaluation during training
    finish_early = False #model is not None

    if model is None:
        if args.llava_checkpoint is not None:
            pretrained = eval_args.llava_checkpoint
        tokenizer, model, image_processor, _ = prepare_llava(pretrained)   
       
    device = torch.device(f'cuda:{rank}') 

    global_avion_correct = torch.tensor(0.0, device=device)
    global_running_corrects = torch.tensor(0.0, device=device)
    global_total_samples = torch.tensor(0.0, device=device)


    for idx, (frames, mc_data, time_meta, global_index) in tqdm(enumerate(val_dataloader)):        

        with torch.no_grad():
            global_index = global_index[0]
            mc_data = mc_data[0]
            time_meta = time_meta[0]         
            gt_name = mc_data['gt_answer_name'][0]
            local_avion_correct = torch.tensor(0.0, device=device)
            local_running_corrects = torch.tensor(0.0, device=device)
            local_total_samples = torch.tensor(0.0, device=device)            
                
            if eval_args.action_predictions:
                avion_pred = mc_data['avion_pred']
                if gt_name == avion_pred:               
                    local_avion_correct.add_(1)
                    global_avion_correct.add_(1)
                

            # we don't want to evaluate the whole thing
            # let's evaluate 1000 samples to get the complete picture       
            if finish_early and idx> (1000 / dist.get_world_size()):
                break                     
        
            # Update running corrects and total samples
            temperature = 0
            ensemble_k = 1

            if eval_args.ensemble_test:
                temperature = 1
                ensemble_k = 3

            llava_correct, llava_pred = ensemble_llava_evaluation(
                                                        eval_args.pretrained_name,
                                                        gt_name,
                                                        frames, 
                                                        tokenizer,
                                                        model,
                                                        image_processor,
                                                        mc_data,
                                                        eval_args.clip_length,
                                                        eval_args.llava_num_frames,
                                                        temperature = temperature,
                                                        ensemble_k = ensemble_k,
                                                        time_meta = time_meta)
                                                        

            # log the predictions into prediciton analysis
        
            val_dataset.prediction_analysis.log(global_index,
                                                llava_pred,
                                                gt_name,
                                                mc_data['all_avion_preds'],
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
    dist.all_reduce(global_total_samples, op=dist.ReduceOp.SUM)
    if eval_args.action_predictions:
        dist.all_reduce(global_avion_correct, op=dist.ReduceOp.SUM)

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
    
    evaluate_on_EK100(args)
   
