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
from action.utils import generate_label_map

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
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res


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
        self.nouns = self.nouns_df['key'].to_list()
        self.verbs = self.verb_df['key'].to_list()

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
        if self.dataset == 'ego4d':
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames = video_loader(self.root, vid, 'mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if isinstance(narration, list):
                if narration_selection == 'random':
                    narration = random.choice(narration)
                elif narration_selection == 'concat':
                    narration = '. '.join(narration)
                elif narration_selection == 'list':
                    pass
                else:
                    raise ValueError
            return frames, narration
        elif self.dataset == 'ek100_mir':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos])
            else:
                return frames, (narration, 1)
        elif self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            # chunk length is the chunked video clip length
            # clip length is number of frames we want to sample from the clip
            frames = video_loader(self.root, vid_path, 'MP4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            return frames, '{}:{}'.format(verb, noun)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 chunk_len=300,
                 clip_length=32, clip_stride=2,
                 threads=1,
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False,
                 rcc_params=(224,),
                 subsample_stride=None):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        frames, caption = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
        )

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)[0]

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoClassyDataset(VideoCaptionDatasetBase):
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
        is_trimmed=True):
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

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
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

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label


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
        topk_predictions = 5    
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
        self.valid_gts = []
        self.eval_args = eval_args
        for verb in self.verbs:
            for noun in self.nouns:
                self.valid_gts.append(f'{verb} {noun}')
        self.labels = labels
        self.topk_predictions = topk_predictions
        
    def __getitem__(self, i):
        frames, label = self.get_raw_item(
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

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)
        
        verb, noun = label.split(':')
        verb, noun = self.verbs[int(verb)], self.nouns[int(noun)]

        noun = noun.replace(':', ' ')
        
        letters = [chr(65+i) for i in range(26)][:self.topk_predictions]
        options = list(range(26))[:self.topk_predictions]
        option_names = []

        # randomly sample topk actions from valid gts
        
        wrong_answer_indices = np.random.choice(len(self.valid_gts), size = self.eval_args.topk_predictions, replace = False)        
        wrong_answers = [self.valid_gts[index] for index in wrong_answer_indices]
        
        for i in range(len(wrong_answers)):
            options[i] =  f'{letters[i]}. {wrong_answers[i]}'
            option_names.append(wrong_answers[i])

        # correct answer must come from the available letters
        
        correct_answer_index =  np.random.choice(len(wrong_answers), size=1, replace=False)[0]        
        correct_answer_letter = letters[correct_answer_index]

        option_names[correct_answer_index] = f'{verb} {noun}'        
        options[correct_answer_index] = f'{correct_answer_letter}. {verb} {noun}'

        data = {
            'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer'},
            'option': {0: options},
            # the correct letter in mc
            'answer': {0: correct_answer_letter},
            # for inspecting
            'answer_name': {0: f'{verb} {noun}'}
        }
       
        return frames, data, option_names 


def get_downstream_dataset(transform, crop_size, eval_args, subset='train', label_mapping=None, labels = None):
    
    if subset == 'val':
        return VideoMultiChoiceDataset(
            eval_args.dataset, eval_args.root, eval_args.val_metadata, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=eval_args.num_clips,
            chunk_len=eval_args.video_chunk_length,
            clip_length=eval_args.clip_length, clip_stride=eval_args.clip_stride,
            threads=eval_args.decode_threads,
            fast_rcc=eval_args.fused_decode_crop, rcc_params=(crop_size, ),
            is_trimmed=not eval_args.dataset == 'charades_ego',
            labels = labels,
            eval_args = eval_args,
            topk_predictions = eval_args.topk_predictions
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")


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
    parser.add_argument('--llm_size', default='7b', type=str, help='llm size')
    parser.add_argument('--llava_num_frames', default=16, type=int, help='number of frames for llava')
    ## avaion refinement 
    parser.add_argument('--action_predictions', default=None, type=str, help='path to action predictions')
    parser.add_argument('--topk_predictions', default = 5, type =int)
    parser.add_argument('--llava_checkpoint', default = None, type = str)
    
    return parser

def prepare_llava(pretrained):

    import warnings
    from llava.model.builder import load_pretrained_model    
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    model_name = "llava_qwen"

    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

    return tokenizer, model, image_processor, max_length


def get_topk_predictions(data, idx,  k):

    letters = [chr(65+i) for i in range(26)][:k]
    options = list(range(26))[:k]

    predictions = data[str(idx)]['predictions'][:k]
    target =  data[str(idx)]['target']
    
    predictions = [e.replace(':', ' ') for e in predictions]
    
    for i in range(len(options)):
        options[i] = f'{letters[i]}. {predictions[i]}'
                
    mc_data = {
        'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer. The letter is as A, B, C, ..'},
        'option': {0: options}        
        }    

    return mc_data, predictions, target    

def evaluate_on_EK100(eval_args, model= None, tokenizer= None, max_length= None, image_processor= None):

    if image_processor is None:
        image_processor = model.get_vision_tower().image_processor

    gpu_val_transform_ls = []

    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    crop_size = 336

    labels, mapping_vn2act, _, _ = generate_label_map(Path(eval_args.val_metadata).parent) 
    val_dataset = get_downstream_dataset(
        val_transform_gpu, crop_size, eval_args, subset='val', label_mapping=mapping_vn2act,
        labels = labels
    )

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) 

    gts = []
    preds = []
    running_corrects = 0
    total_samples = 0

    if eval_args.action_predictions:
        valid_letters = [chr(65+i) for i in range(26)][:eval_args.topk_predictions]
    else:
        valid_letters = ['A', 'B', 'C', 'D', 'E']

    if not eval_args.action_predictions:        
        log_filename = f'llava_ov_{eval_args.llava_num_frames}f_{eval_args.llm_size}.log'
    else:
        log_filename = f'llava_ov_{eval_args.llava_num_frames}f_{eval_args.llm_size}_action_{eval_args.topk_predictions}.log'
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Set the same format for console handler as well
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    
    logger = logging.getLogger(__name__)

    pretrained = f"lmms-lab/llava-onevision-qwen2-{eval_args.llm_size}-ov"

    # so we know it's evaluation during training
    finish_early = model is not None

    if model is None:
        if hasattr(eval_args, "llava_checkpoint"):
            pretrained = eval_args.llava_checkpoint
        tokenizer, model, image_processor, max_length = prepare_llava(pretrained)   

    if eval_args.action_predictions:
        with open(eval_args.action_predictions, 'r') as f:
            predictions = json.load(f)
        
    avaion_correct = 0    
    
    for idx, (frames, mc_data, option_names) in tqdm(enumerate(val_dataloader)):
        gt = mc_data['answer'][0][0]
        gt_name = mc_data['answer_name'][0][0]        
                
        if eval_args.action_predictions:
            mc_data, avaion_pred, target = get_topk_predictions(predictions, idx, eval_args.topk_predictions)
            target = target.replace(':', ' ')
            if target == avaion_pred[0]:
                avaion_correct+=1
        # we don't want to evaluate the whole thing
        if finish_early and idx>9:
            break
        
        pred = llava_inference(frames, tokenizer, model, image_processor, max_length, mc_data,  clip_length = eval_args.clip_length, num_frames=eval_args.llava_num_frames)
        
        # if valid letter is found in the prediction, then we will use that as the prediction
        found = False
        rank0_print ('llava pred', pred)
        rank0_print ('gt_name', gt_name)
        for letter in valid_letters:
            if letter in pred:
                pred = letter
                found = True
                break
        if not found:
            pred = 'N/A'

        if eval_args.action_predictions:
            if found:
                pred_index = valid_letters.index(pred)
                pred_name = avaion_pred[pred_index]
                
            else:
                pred_name = 'N/A'                
        else:
            if found:
                pred_index = valid_letters.index(pred)            
                pred_name = option_names[pred_index][0]
            else:
                pred_name = 'N/A'
        gts.append(gt_name)                 
        preds.append(pred_name)

        # Update running corrects and total samples
        running_corrects += (pred_name == target)
        total_samples += 1

        # Calculate and log running mean accuracy
        running_accuracy = running_corrects / total_samples

        logger.info(f'Running accuracy after {total_samples} samples: {running_accuracy:.4f}')

        if eval_args.action_predictions:
            avaion_accuracy = avaion_correct / total_samples
            logger.info(f'Running avaion accuracy after {total_samples} samples: {avaion_accuracy:.4f}')        
        
        
    gts = np.array(gts)
    preds = np.array(preds)
    # get final accuracy 
    accuracy = np.mean(gts == preds)
    logger.info(f'Final accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    evaluate_on_EK100(args)
   
