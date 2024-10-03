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

        letters = [chr(65+i) for i in range(26)][:self.topk_predictions]
        options = list(range(26))[:self.topk_predictions]
        
        
        wrong_answer_indices = np.random.choice(len(self.valid_gts), size = 5, replace = False)
        wrong_answers = [self.valid_gts[index] for index in wrong_answer_indices]
        for i in range(len(wrong_answers)):
            options[i] =  f'{letters[i]}. {wrong_answers[i]}'
            
        correct_answer_index =  np.random.choice(len(letters), size=1, replace=False)[0]
        correct_answer_letter = letters[correct_answer_index]

        options[correct_answer_index] = f'{correct_answer_letter}. {verb} {noun}'

        data = {
            'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer'},
            'option': {0: options},
            'answer': {0: correct_answer_letter},
            'answer_name': {0: f'{verb} {noun}'}
        }
       
        return frames, data      


def get_downstream_dataset(transform, crop_size, args, subset='train', label_mapping=None, labels = None):
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.train_metadata, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
        )
    elif subset == 'val':
        return VideoMultiChoiceDataset(
            args.dataset, args.root, args.val_metadata, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size, ),
            is_trimmed=not args.dataset == 'charades_ego',
            labels = labels,
            topk_predictions = args.topk_predictions
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")


def generate_label_map(args):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    anno_root = Path(args.train_metadata).parent
    for f in [      
        anno_root / 'EPIC_100_train.csv',
        anno_root / 'EPIC_100_validation.csv',
    ]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip the header
        for row in csv_reader:
            
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            narration = row[8]
            if vn not in vn_list:
                vn_list.append(vn)
            if vn not in mapping_vn2narration:
                mapping_vn2narration[vn] = [narration]
            else:
                mapping_vn2narration[vn].append(narration)
            # mapping_vn2narration[vn] = [narration]
    vn_list = sorted(vn_list)
    print('# of action= {}'.format(len(vn_list)))
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}

    labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
    print(labels[:5])    
    return labels, mapping_vn2act


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
    # model
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument('--use-fast-conv1', action='store_true', dest='use_fast_conv1')
    parser.add_argument('--disable-fast-conv1', action='store_false', dest='use_fast_conv1')
    parser.set_defaults(use_fast_conv1=False)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument('--patch-dropout', default=0., type=float)
    parser.add_argument('--drop-path-rate', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float, help='dropout for the last linear layer')
    parser.add_argument('--num-classes', default=3806, type=float, help='number of classes for the last linear layer')
    parser.add_argument('--pretrain-model', default='', type=str, help='path of pretrained model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # mixup
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # training
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=2, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='sgd', choices=['adamw', 'sgd'], type=str)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=4e-05, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--pickle-filename', default='', type=str, help='pickle filename to dump')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    # llava related
    # llm size is type of string and can only be '7b' or '5b' etc.
    parser.add_argument('--llm_size', default='7b', type=str, help='llm size')
    parser.add_argument('--llava_num_frames', default=16, type=int, help='number of frames for llava')
    ## avaion refinement 
    parser.add_argument('--action_predictions', default=None, type=str, help='path to action predictions')
    parser.add_argument('--topk_predictions', default = 5, type =int)

    return parser

def prepare_llava():

    import warnings
    from llava.model.builder import load_pretrained_model    
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    #pretrained = f"lmms-lab/llava-onevision-qwen2-{llm_size}-ov"
    model_name = "llava_qwen"

    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

    return tokenizer, model, image_processor, max_length


def get_topk_predictions(prediction_file, idx,  k):

    with open(prediction_file, 'r') as f:
        data = json.load(f)

    letters = [chr(65+i) for i in range(26)][:k]
    options = list(range(26))[:k]
    
    predictions = data[str(idx)]['predictions'][:k]

    for i in range(len(options)):
        options[i] = f'{letters[i]}. {predictions[i]}'
            
    
    mc_data = {
        'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer'},
        'option': {0: options}        
        }
    return mc_data
    

if __name__ == '__main__':
    from moviepy.editor import ImageSequenceClip
    import torchvision
    import torchvision.transforms._transforms_video as transforms_video
    
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    parser = argparse.ArgumentParser(description='AVION finetune ek100 cls', add_help=False)
    gpu_val_transform_ls = []
    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)
    crop_size = 336

    labels, mapping_vn2act = generate_label_map(args) 
    val_dataset = get_downstream_dataset(
        val_transform_gpu, crop_size, args, subset='val', label_mapping=mapping_vn2act,
        labels = labels
    )

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) 

    gts = []
    preds = []
    running_corrects = 0
    total_samples = 0

    if args.action_predictions:
        valid_letters = [chr(65+i) for i in range(26)][:args.topk_predictions]
    else:
        valid_letters = ['A', 'B', 'C', 'D', 'E']

    if not args.action_predictions:        
        log_filename = f'llava_ov_{args.llava_num_frames}f_{args.llm_size}.log'
    else:
        log_filename = f'llava_ov_{args.llava_num_frames}f_{args.llm_size}_action_{args.topk_predictions}.log'
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Set the same format for console handler as well
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    
    logger = logging.getLogger(__name__)

    pretrained = f"lmms-lab/llava-onevision-qwen2-{args.llm_size}-ov"

    tokenizer, model, image_processor, max_length = prepare_llava()
    
    for idx, (frames, mc_data) in tqdm(enumerate(val_dataloader)):

        gt = mc_data['answer'][0][0]
        
        gts.append(gt)
        
        if args.action_predictions:
            mc_data = get_topk_predictions(args.action_predictions, idx, args.topk_predictions)

        
        pred = llava_inference(frames, tokenizer, model, image_processor, max_length, mc_data,  num_frames=args.llava_num_frames)

        # if valid letter is found in the prediction, then we will use that as the prediction
        found = False
        
        for letter in valid_letters:
            if letter in pred:
                pred = letter
                found = True
                break
        if not found:
            pred = 'N/A'


        preds.append(pred)

        # Update running corrects and total samples
        running_corrects += (pred == gt)
        total_samples += 1

        # Calculate and log running mean accuracy
        running_accuracy = running_corrects / total_samples
        logger.info(f'Running accuracy after {total_samples} samples: {running_accuracy:.4f}')

    gts = np.array(gts)
    preds = np.array(preds)
    # get final accuracy 
    accuracy = np.mean(gts == preds)
    logger.info(f'Final accuracy: {accuracy:.4f}')
   
