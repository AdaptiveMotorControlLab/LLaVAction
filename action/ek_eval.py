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
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
import os
sys.path[0] = os.path.dirname(sys.path[0])
from action.llava_ov_inference import llava_inference
import json
import logging
from llava.utils import rank0_print
from action.utils import generate_label_map, MultiChoiceGenerator, match_answer, parse_avion_predictions
import copy
from collections import Counter 

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0'
    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, '{}.{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()

        return torch.from_numpy(frames.astype(np.float32))

    else:
        time_meta = {}
        
        time_meta['duration'] = end_second - second
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        while True:
            video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
            
            if not osp.exists(video_filename):
                # print("{} does not exists!".format(video_filename))
                chunk_end -= chunk_len
            else:
                vr = decord.VideoReader(video_filename)
                end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                assert chunk_start <= chunk_end
                break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        all_frames = []
        all_frame_ids = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            all_frame_ids.append(frame_ids)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        time_meta['n_frames'] = res.shape[0]
        all_frame_ids = np.concatenate(all_frame_ids, axis = 0)
        frame_time = [e/fps for e in all_frame_ids]
        frame_time = [f"{i:.2f}s" for i in frame_time]
        time_meta['frame_time'] = frame_time
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res, time_meta


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
            frames, time_meta = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
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
        
        return frames, data, time_meta



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
    ## avaion refinement 
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

    print ('overwrite config', overwrite_config)
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
        rank0_print ('generated new option sequence')
        rank0_print (options)

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

    return match_answer(counter.most_common(1)[0][0], gt_name)



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

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) 

    running_corrects = 0
    total_samples = 0    
        
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
    finish_early = model is not None

    if model is None:
        if args.llava_checkpoint is not None:
            pretrained = eval_args.llava_checkpoint
        tokenizer, model, image_processor, _ = prepare_llava(pretrained)   

    if eval_args.action_predictions:
        with open(eval_args.action_predictions, 'r') as f:
            predictions = json.load(f)
        
    avaion_correct = 0    
    
    for idx, (frames, mc_data, time_meta) in tqdm(enumerate(val_dataloader)):

        gt_name = mc_data['gt_answer_name'][0][0]
                
        if eval_args.action_predictions:
            mc_data = get_topk_predictions(predictions, idx, eval_args.topk_predictions)
            avion_pred = mc_data['avion_pred']
            if gt_name == avion_pred:
                avaion_correct+=1

        # we don't want to evaluate the whole thing
        # let's evaluate 1000 samples to get the complete picture
        if finish_early and idx>999:
            break                     

        # Update running corrects and total samples
        running_corrects += ensemble_llava_evaluation(
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
                                                              
        total_samples += 1

        # Calculate and log running mean accuracy
        running_accuracy = running_corrects / total_samples

        logger.info(f'running accuracy: {running_accuracy:.4f}')
        if eval_args.action_predictions:
            avaion_accuracy = avaion_correct / total_samples

        
    logger.info(f'Running avaion accuracy after {total_samples} samples: {avaion_accuracy:.4f}')         
    logger.info(f'Final accuracy: {running_accuracy:.4f}')
    return running_accuracy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    evaluate_on_EK100(args)
   
