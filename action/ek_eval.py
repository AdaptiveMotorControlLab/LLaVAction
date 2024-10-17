import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import argparse
import decord
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
import sys
import os
sys.path[0] = os.path.dirname(sys.path[0])
from action.llava_ov_inference import llava_inference
import json
import logging
from llava.utils import rank0_print
from action.utils import generate_label_map, MultiChoiceGenerator, match_answer, parse_avion_predictions, avion_video_loader
from action.prediction_analysis import PredictionAnalysis
import copy
from collections import Counter 
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(backend='nccl')
rank = dist.get_rank()
torch.cuda.set_device(rank)

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed
        anno_root = Path(metadata).parent
        self.verb_file = str(anno_root / 'EPIC_100_verb_classes.csv')
        self.noun_file = str(anno_root / 'EPIC_100_noun_classes.csv')
        self.verb_df = pd.read_csv(self.verb_file)
        self.nouns_df = pd.read_csv(self.noun_file)     

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            video_list = glob.glob(osp.join(self.root, '*/*.MP4'))
            fps_dict = {video: decord.VideoReader(video + '/0.MP4').get_avg_fps() for video in video_list}
            # all becoming fps 30
            # metadata is the annotation file. 
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    # add verbs and nouns
                   
                    vid_path = '{}/{}'.format(pid, vid)
                    fps = fps_dict[osp.join(self.root, vid_path + '.MP4')]
                    # start_frame = int(np.round(fps * start_timestamp))
                    # end_frame = int(np.ceil(fps * end_timestamp))
                    # verb and noun here mean verb noun classes respectively
                    # narration is basically verb + noun
                    self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun))

            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1
        else:
            raise NotImplementedError

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):
        if self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            # chunk length is the chunked video clip length
            # clip length is number of frames we want to sample from the clip
            frames, time_meta = avion_video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            time_meta['start_second'] = start_second
            time_meta['end_second'] = end_second
            time_meta['fps'] = fps
            time_meta['vid_path'] = vid_path
            return frames, '{}:{}'.format(verb, noun), time_meta
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoMultiChoiceDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        labels = None,
        is_trimmed=True,
        eval_args = None,
        topk_predictions = 5,
        verb_maps = None,
        noun_maps = None
    ):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample
        self.eval_args = eval_args
        self.verb_maps = verb_maps
        self.noun_maps = noun_maps
        self.vn_list = list(self.label_mapping.keys())        

        self.labels = labels
        self.topk_predictions = topk_predictions
        self.ann_root = Path(metadata).parent
        self.mc_generator = MultiChoiceGenerator(self.ann_root)
        self.rank = dist.get_rank()
        self.prediction_analysis = PredictionAnalysis(rank = self.rank)
        
    def __getitem__(self, i):
        frames, label, time_meta = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # for llava-video to work, we also need time meta data.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)
        
        data = self.mc_generator.generate_multi_choice(label, self.topk_predictions)
        
        return frames, data, time_meta, i



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
    # llm size is type of string and can only be '7b' or '5b' etc.
    parser.add_argument('--pretrained_name', default = '', type = str, help ='the name in huggingface')
    parser.add_argument('--llava_num_frames', default=16, type=int, help='number of frames for llava')
    ## avion refinement 
    parser.add_argument('--action_predictions', default=None, type=str, help='path to action predictions')
    parser.add_argument('--topk_predictions', default = 5, type =int)
    parser.add_argument('--llava_checkpoint', default = None, type = str)

    
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

def get_topk_predictions(data, idx, k):

    letters = [chr(65+i) for i in range(26)][:k]
    options = list(range(26))[:k]

    predictions = data[str(idx)]['predictions'][:k]
    predictions = parse_avion_predictions(predictions)    

    for i in range(len(options)):              
        options[i] = f'{letters[i]}. {predictions[i]}'
                
    mc_data = {
        'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer.'},
        'options': {0: options},
        'valid_letters': letters,
        'avion_pred': predictions[0]
        }    
    
    return mc_data

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
                              is_test = False
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
                            is_test = is_test,
                            time_meta = time_meta
                               )
        
        rank0_print ('llava pred', pred, 'avion_pred', avion_pred, 'gt_name', gt_name) 
        if '.' in pred:
            sep = pred.index('.')
            pred = pred[sep+1:].strip()
        preds.append(pred)
        
    counter = Counter(preds)
    rank0_print ('inspecting the counter', counter)
    rank0_print ('most common', counter.most_common(1)[0][0])

    return match_answer(counter.most_common(1)[0][0], gt_name), counter.most_common(1)[0][0]



def evaluate_on_EK100(eval_args, 
                      model= None, 
                      tokenizer= None, 
                      image_processor= None):

    if model is not None:
        image_processor = model.get_vision_tower().image_processor

    gpu_val_transform_ls = []
    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)
    crop_size = 336
    labels, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(eval_args.val_metadata).parent) 

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

            )

    if dist.is_initialized():
        sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler = None

    val_dataloader = DataLoader(val_dataset, sampler = sampler, batch_size=1, shuffle=False)    
        
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

    if eval_args.action_predictions:
        with open(eval_args.action_predictions, 'r') as f:
            predictions = json.load(f)        

    device = torch.device(f'cuda:{rank}') 

    global_avion_correct = torch.tensor(0.0, device=device)
    global_running_corrects = torch.tensor(0.0, device=device)
    global_total_samples = torch.tensor(0.0, device=device)


    for idx, (frames, mc_data, time_meta, global_index) in tqdm(enumerate(val_dataloader)):        

        with torch.no_grad():
            global_index = global_index.item()

            gt_name = mc_data['gt_answer_name'][0][0]
            local_avion_correct = torch.tensor(0.0, device=device)
            local_running_corrects = torch.tensor(0.0, device=device)
            local_total_samples = torch.tensor(0.0, device=device)
                
            if eval_args.action_predictions:
                mc_data = get_topk_predictions(predictions, global_index, eval_args.topk_predictions)
                avion_pred = mc_data['avion_pred']
                if gt_name == avion_pred:
                    local_avion_correct.add_(1)
                    global_avion_correct.add_(1)

            # we don't want to evaluate the whole thing
            # let's evaluate 1000 samples to get the complete picture       
            if finish_early and idx> (1000 / dist.get_world_size()):
                break                     
        
            # Update running corrects and total samples
            
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
                                                        temperature = 0,
                                                        ensemble_k = 1,
                                                        time_meta = time_meta,
                                                        is_test = not finish_early)

            # log the predictions into prediciton analysis
        
            val_dataset.prediction_analysis.log(global_index,
                                                llava_pred,
                                                gt_name,
                                                predictions[str(global_index)],
                                                time_meta['start_second'].item(),
                                                time_meta['end_second'].item(),
                                                time_meta['vid_path'],
                                                dataset_name = 'EK100')

        


            local_running_corrects.add_(llava_correct)
            global_running_corrects.add_(llava_correct)
                                                                
            local_total_samples.add_(1)
            global_total_samples.add_(1)

            logger.info(f'Process {dist.get_rank()} - local_total_samples: {local_total_samples:.4f}')

            logger.info(f'Process {dist.get_rank()} - loca_llava_correct: {llava_correct:.4f}')

            logger.info(f'Process {dist.get_rank()} - local_running_corrects: {local_running_corrects:.4f}')


        # Calculate and log running mean accuracy
        # dist.barrier()
        # dist.all_reduce(local_running_corrects, op=dist.ReduceOp.SUM)
        # dist.all_reduce(local_total_samples, op=dist.ReduceOp.SUM)
        # if eval_args.action_predictions:
        #     dist.all_reduce(local_avion_correct, op=dist.ReduceOp.SUM)
        # dist.barrier()
        # # Calculate global accuracy after reduction
        # local_running_accuracy = local_running_corrects.item() / local_total_samples.item()
        # local_avion_accuracy = local_avion_correct.item() / local_total_samples.item()

        # logger.info(f'Process {dist.get_rank()} - Running accuracy: {local_running_accuracy:.4f}')
        # logger.info(f'Process {dist.get_rank()} - AvionRunning accuracy: {local_avion_accuracy:.4f}')

    

    dist.barrier()
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
   
