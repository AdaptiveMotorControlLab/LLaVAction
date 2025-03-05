import csv
import glob
import os.path as osp
import pickle
import pandas as pd
import torch
import decord
from pathlib import Path
from llavaction.action.utils import  AvionMultiChoiceGenerator,  RandomMultiChoiceGenerator, avion_video_loader, EK100_frame_loader
from llavaction.action.prediction_analysis import PredictionAnalysis
import torch.distributed as dist

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
                    start_timestamp, end_timestamp = round(datetime2sec(row[4]),2), round(datetime2sec(row[5]),2)
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
        elif self.dataset in ['ekframes_cls']:
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    start_frame, end_frame = int(row[6]), int(row[7])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    fps = end_frame / end_timestamp

                    vid_path = '{}/{}'.format(pid, vid)

                    self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun, start_frame, end_frame, vid))
            aa = 1
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
        elif self.dataset == 'ekframes_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun, start_frame, end_frame, vid = self.samples[i]
            video_file = osp.join(self.root, vid)
            frames, time_meta = EK100_frame_loader(video_file, start_frame, end_frame, start_second, end_second, 
                                                   clip_length = clip_length, jitter = is_training)
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
        noun_maps = None,
        eval_result_folder = None,
        gen_type = 'action_model',
        action_representation = 'GT_random_narration_cut',
        mapping_vn2narration = None,
        avion_predictions = None,
        n_narrations = -1,
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
        self.gen_type = gen_type
        if gen_type == 'action_model':
            self.mc_generator = AvionMultiChoiceGenerator(self.ann_root)
        elif gen_type == 'random':
            self.mc_generator = RandomMultiChoiceGenerator(self.ann_root)
        self.rank = dist.get_rank()
        self.prediction_analysis = PredictionAnalysis(rank = self.rank, save_folder = eval_result_folder)
        self.action_representation = action_representation
        self.n_narrations = n_narrations
        self.mapping_vn2narration = mapping_vn2narration
        self.avion_predictions = avion_predictions
        
        
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
        narration = self.samples[i][4]
        avion_preds = self.avion_predictions[str(i)]['predictions']
        if self.gen_type =='action_model':
            data = self.mc_generator.generate_multi_choice(label, 
                                                            avion_preds,                                                       
                                                            narration,
                                                            self.topk_predictions, 
                                                            self.action_representation,
                                                            self.n_narrations,
                                                            self.labels,
                                                            self.mapping_vn2narration,                                                        
                                                            self.verb_maps, 
                                                            self.noun_maps,
                                                            benchmark_testing = self.eval_args.benchmark_testing,
                                                            is_train = False) # note we only use this dataset for evaluation for now.
        else:
            data = self.mc_generator.generate_multi_choice(label, 
                                                            narration,
                                                            self.topk_predictions, 
                                                            self.action_representation,
                                                            self.n_narrations,
                                                            self.labels,
                                                            self.mapping_vn2narration,                                                        
                                                            self.verb_maps, 
                                                            self.noun_maps,
                                                            benchmark_testing = self.eval_args.benchmark_testing,
                                                            is_train = False) # no
            
       
        return frames, data, time_meta, i
