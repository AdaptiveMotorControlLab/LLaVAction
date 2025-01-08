import csv 
import numpy as np
import random
import os
import decord
import os.path as osp
import torch
import pandas as pd
import ast
# import inflect
import copy
from tqdm import tqdm
from collections import Counter
import pickle
from PIL import Image, ImageFile
import cv2
from llava.action.render_utils import render_frame


def remove_sub_nouns(nlp, narration, verb, nouns, cache_file = None):
    narration = copy.deepcopy(narration)
    noun_list = ast.literal_eval(nouns)
    if len(noun_list) > 0:
        v_words = verb.split('-')
        n_words = noun_list[0].split(':')
        n_words = n_words[1:] + [n_words[0]]

        # deal with some special cases
        if 'leaf' in n_words and 'leaves' in narration:
            # replace the word 'leaf' with 'leaves'
            n_words[n_words.index('leaf')] = 'leaves'
        if 'continue ' in narration:
            # remove the word 'continue' in the narration
            narration = narration.replace('continue ', '')
        if 'something' in narration:
            narration = narration.replace('something', ' '.join(n_words))

        words = copy.deepcopy(v_words + n_words)
        narration_words = narration.split(' ')
        # new_narration_words = [inflect_tool.singular_noun(word) or word for word in narration_words]
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                doc = pickle.load(f)
        else:
            doc = nlp(narration)
            with open(cache_file, "wb") as f:
                pickle.dump(doc, f)
        new_narration_words = [token.lemma_ for token in doc]
        keep_words = []
        for word, new_word in zip(narration_words, new_narration_words):
            if word in words:
                keep_words.append(word)
                words.remove(word)
            elif new_word in words:
                keep_words.append(new_word)
                words.remove(new_word)
        new_narration = ' '.join(keep_words)
        # assert len(words) == 0

        # deal with some special cases
        if len(words) != 0:
            keep_words = []
            verb_added = False
            noun_added = False
            for word, new_word in zip(narration_words, new_narration_words):
                if word in v_words:
                    keep_words.append(word)
                    verb_added = True
                elif new_word in v_words:
                    keep_words.append(new_word)
                    verb_added = True
                elif (word in n_words or new_word in n_words) and not noun_added:
                    keep_words.append(' '.join(n_words))
                    noun_added = True
            if not verb_added:
                keep_words = [' '.join(v_words)] + keep_words
            if not noun_added:
                keep_words.append(' '.join(n_words))
            new_narration = ' '.join(keep_words)
        
        # # debug
        # if new_narration == 'pick up teaspoon honey' or new_narration == 'pick up measure' \
        #         or new_narration == 'remove tap on' or new_narration == 'pick up glass water' \
        #         or new_narration == 'pick up washing up brush' or new_narration == 'pick up glass water':
        #     aa = 1
         
    else:
        new_narration = narration
        
    return new_narration


def remove_option_letter(answer):
    if '. ' in answer:
        return answer.split('. ')[1]
    else:
        return answer

def generate_label_map(anno_root, action_representation, cache_file = None):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}

    noun_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_noun_classes_v2.csv'))
    verb_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_verb_classes.csv'))
    # from id to name
    verb_maps = {} if 'key' in action_representation or action_representation == 'first_sample' else None
    noun_maps = {} if 'key' in action_representation or action_representation == 'first_sample' else None
    if 'key' in action_representation:
        # use the id in noun_classes_pd and verb_classes_pd as the key, use the key in noun_classes_pd and verb_classes_pd as the value
        for i, row in verb_classes_pd.iterrows():
            verb_maps[str(row['id'])] = row['key']
        for i, row in noun_classes_pd.iterrows():
            elements = row['key'].split(':')
            if len(elements) == 1:
                noun_maps[str(row['id'])] = row['key']
            else:
                noun_maps[str(row['id'])] = ' '.join(elements[1:] + [elements[0]]) # this is to refact the noun like 'machine:sous:vide'

    # inflect_tool = inflect.engine()
    
    for f in [      
        os.path.join(anno_root,'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip the header
        for row in tqdm(csv_reader):
            
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            if action_representation == 'first_sample':
                if row[10] not in verb_maps.keys():
                    verb_maps[row[10]] = row[9]
                if row[12] not in noun_maps.keys():
                    noun_maps[row[12]] = row[11]

            if vn not in vn_list:
                vn_list.append(vn)

            narration = row[8]
            if 'cut' in action_representation:
                import spacy
                nlp = spacy.load('en_core_web_sm')
                narration = remove_sub_nouns(nlp, narration, row[9], row[13], cache_file = cache_file)
                
            if vn not in mapping_vn2narration:
                mapping_vn2narration[vn] = [narration]
            else:
                mapping_vn2narration[vn].append(narration)
            # mapping_vn2narration[vn] = [narration]
    vn_list = sorted(vn_list)
    print('# of action= {}'.format(len(vn_list)))
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}

    # labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
    labels = {}
    for vn, narrations in mapping_vn2narration.items():
        frequency_count = Counter(narrations)
        sorted_unique_list = [item for item, count in frequency_count.most_common()]
        labels[vn] = sorted_unique_list

    return labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps



def format_task_related_prompt(question, question_type, perspective = "first_person"):
    """
    Task related prompt is impacted by the question_type.
    We currently support mc_{action_representation} and gpt-gt-reason
    We are thinking about tweaking the prompt based on the action representation.
    """
    if perspective == "first_person":
        perspective_prefix = "You are seeing this video from egocentric view and you are the person. Your hands are sometimes interacting with obects. What action are you performing? "
    elif perspective == "third_person":
        perspective_prefix = "The video is taken from egocentric view. What action is the person performing? "
    if question_type.startswith("mc_"):
        action_rep_suffix = "Given multiple choices, format your answer briefly such as 'A. move knife'. "              
        prefix = f"{perspective_prefix}{action_rep_suffix}\n"
        assert isinstance(question, list)
        suffix = ", ".join(question)
        suffix = "Here are the options of actions you are selecting:\n" + suffix 
        ret = prefix + suffix
    elif question_type == "gpt-gt-reason":
        ret = f"{perspective_prefix}Describe in details what you see from the video frames."
    
    elif question_type == "validation":
        ret = f"Ask yourself questions to validate your notes."
    
    elif question_type == "gpt-gt-strong-reason":
        ret = f"{perspective_prefix} Describe in details what you see and answer the multi-choice question. Explain why wrong answers are wrong and why the correct answer is correct. "
        suffix = ", ".join(question)
        suffix = "Here are the options of actions you are selecting:\n" + suffix  
        ret = ret + suffix       

    elif question_type == "dpo":
        ret = "You are seeing this video from egocentric view and you are the person. Your hands are sometimes interacting with obects. Describe in details what you see and what you are doing."

    elif question_type == "gpt-gt-instruct-reason":
        ret = question
    elif question_type == "gpt-hand-object":
        ret = question
    elif question_type == "cot_mc":
        """
        Explain the reasoning first and do the multiple-choice.        
        """
        action_rep_suffix = "Describe what you see in details. Afterwards, briefly format your answer such as 'A. move knife'. "              
        prefix = f"{perspective_prefix} {action_rep_suffix}\n"
        assert isinstance(question, list)
        suffix = ", ".join(question)  
        suffix = "Here are the options of choices you are selecting:\n" + suffix 
        ret = prefix + suffix  
    else:
        raise NotImplementedError(f"question_type: {question_type} is not supported")      
        

    return ret

def format_time_instruction(video_duration, n_frames, include_frame_time = False):

    prefix = f"The provided video lasts for {video_duration:.3f} seconds, and {n_frames} frames are uniformly sampled from it."

    frame_time = [i * (video_duration / n_frames) for i in range(n_frames)]
    frame_time = ", ".join([f"{i:.2f}s" for i in frame_time])
    
    suffix = ""
    if include_frame_time:
        suffix = f"These frames are located at {frame_time}. The video duration is {video_duration:.3f} seconds. "
    
    return prefix + suffix


def format_llava_prompt(image_token, 
                        question, 
                        video_duration,
                        n_frames,
                        question_type,
                        include_time_instruction = False,
                        include_frame_time = False
                        ):
    """
    baseline llava prompt: {image_token}\n{task_related_prompt}
    with time instruction: {image_token}\n{time_instruction}\n{task_related_prompt}

    """

    task_related_prompt = format_task_related_prompt(question, question_type)

    time_instruction =  format_time_instruction(video_duration, n_frames, include_frame_time)

    if include_time_instruction:
        ret = f"{image_token}\n{time_instruction}{task_related_prompt}"
    else:
        ret = f"{image_token}\n{task_related_prompt}"

    return ret

def match_answer(pred, gt):          
    return pred == gt

def parse_avion_predictions(predictions):
    return [pred.replace(':', ' ', 1) for pred in predictions]   

# DEPRECATED
class MultiChoiceGenerator:
    """
    Generating multi choice
    """
    def __init__(self, ann_root):
        self.ann_root = ann_root
    

    def generate_multi_choice(self, gt_vn, k, verb_maps, noun_maps):

        raise NotImplementedError("This is an abstract class")

        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list

        """        

        # let v_id and n_id be string type
        gt_v_id, gt_n_id = gt_vn.split(':')    
        assert isinstance(gt_v_id, str) and isinstance(gt_n_id, str)
        gt_v_name, gt_n_name = verb_maps[gt_v_id], noun_maps[gt_n_id]

        # letters as A, B, C, D, .. Note we maximally support 26 letters
        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]
        vn_list = list(self.mapping_vn2act.keys())
        action_list = [f"{verb_maps[e.split(':')[0]]} {noun_maps[e.split(':')[1]]}" for e in vn_list]
        wrong_answers = np.random.choice(action_list, size = k-1, replace = False)
        gt_answer = f'{gt_v_name} {gt_n_name}'

        answers = [gt_answer] + list(wrong_answers)
        random.shuffle(answers)

        options = []
        for answer, letter in zip(answers, letters):
            options.append(f'{letter}. {answer}')

        gt_letter = letters[answers.index(gt_answer)]
        data = {
                'options': {0: options},
                # the correct letter in mc
                # for inspecting
                'gt_answer_letter': {0: gt_letter},
                'gt_answer_name': {0: gt_answer},
                'valid_letters': letters,
            }
        
        return data

def parse_vn_ids(answer_id, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps):
    answer_items = []
    if 'key' in action_representation or action_representation == 'first_sample':
        v_id, n_id = answer_id.split(':')
        v_name, n_name = verb_maps[v_id], noun_maps[n_id]
        answer_items.append(f'{v_name} {n_name}')
    if 'random_narration' in action_representation:
        # randomly select a narration from mapping_vn2narration
        answer_items.append(random.choice(mapping_vn2narration[answer_id]))
    elif 'top1_narration' in action_representation:
        # select the top1 narration from labels
        answer_items.append(labels[answer_id][0])
    elif 'topk_narration' in action_representation:
        assert n_narrations > 0
        # select the topk narrations from labels
        answer_items.extend(['example usages could be']+ labels[answer_id][:n_narrations])

    if 'GT' in action_representation and answer_id == gt_vn:
        answer_items = [narration] 

    return ', '.join(answer_items)

class AvionMultiChoiceGenerator(MultiChoiceGenerator):
    """
    Generate multichoice using avion predictions
    """
    def __init__(self, ann_root):
        super().__init__(ann_root)
    

    def train_generate(self, gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps):
        """
        During training, the avion predictions have some randomness from the top 2k.
        One gt is guaranteed to exist in the returned options
        """
        # we should have plenty of predictions to select, so let's not always pick the hardest

        avion_predictions = avion_predictions[:k*2]
        # avion_predictions = parse_avion_predictions(avion_predictions)
        if gt_vn in avion_predictions:
            avion_predictions.remove(gt_vn)       

        # just so that it's not strictly desending with confidence
        random.shuffle(avion_predictions)
        avion_predictions = avion_predictions[:k-1]

        answer_ids = [gt_vn] + avion_predictions

        random.shuffle(answer_ids)

        answers = []
        for answer_id in answer_ids:

            answer = parse_vn_ids(answer_id, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
            answers.append(answer)
        
        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]

        options = []
        for answer, letter in zip(answers, letters):
            options.append(f'{letter}. {answer}')

        gt_letter = letters[answer_ids.index(gt_vn)]
        gt_answer = answers[answer_ids.index(gt_vn)]

        mc_data = {
                'options': {0: options},
                # the correct letter in mc
                # for inspecting
                'gt_answer_letter': {0: gt_letter},
                'gt_answer_name': {0: gt_answer},
                'valid_letters': letters
            }  
        return mc_data              

    def test_generate(self, gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps):
        """
        During testing, we use the top k predictions from avion. No randomness. We do not mix the gt_vn with the avion predictions
        """        

        answer_ids = avion_predictions[:k]
        answers = []
        for answer_id in answer_ids:
            answer = parse_vn_ids(answer_id, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
            answers.append(answer)
        
        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]

        options = []
        for answer, letter in zip(answers, letters):
            options.append(f'{letter}. {answer}')

        # note the gt_answer cannot come from narration, as some action representation turns avion predictions to non-narration format
        gt_answer = parse_vn_ids(gt_vn, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)

        mc_data = {
                'options': {0: options},               
                'gt_answer_name': {0: gt_answer},
                'valid_letters': letters,
                'avion_pred': answers[0],
                'all_avion_preds': answers
            }

        
        
        return mc_data        

    def generate_multi_choice(self, 
                              gt_vn, 
                              avion_predictions, 
                              narration, 
                              k, 
                              action_representation, 
                              n_narrations, 
                              labels, 
                              mapping_vn2narration, 
                              verb_maps, 
                              noun_maps,
                              is_train = True
                              ):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list that is not gt_vn (this is important as avion_predictions can contain correct prediction)        

        """    
        if is_train:
            return self.train_generate(gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
        else:
            return self.test_generate(gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
    
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
    

def avion_video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0' 
    time_meta = {}
    
    time_meta['duration'] = end_second - second

    assert end_second > second, 'end_second should be greater than second'

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
            print('IndexError', root, vid, ext, second, end_second)
        all_frames.append(frames)
        all_frame_ids.append(frame_ids)
        if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
            break
    res = np.concatenate(all_frames, axis=0)
    time_meta['n_frames'] = res.shape[0]
    all_frame_ids = np.concatenate(all_frame_ids, axis = 0)
    frame_time = [e/fps for e in all_frame_ids]
    frame_time-= frame_time[0]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    time_meta['frame_time'] = frame_time
    assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
    return res, time_meta

def EK100_frame_loader(root, start_frame, end_frame, start_second, end_second, clip_length=32, jitter=False):
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=jitter)
    imgs = []
    for frame_id in frame_ids:
        frame_name = osp.join(root, 'frame_{:0>10d}.jpg'.format(frame_id))
        with open(frame_name, "rb") as fp:
            img_bytes = fp.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)
    buffer = np.array(imgs)
    # compute frame time
    time_meta = {}
    time_meta['duration'] = end_second - start_second
    time_meta['n_frames'] = len(imgs)
    fps = (end_frame - start_frame) / (end_second - start_second)
    frame_time = [e/fps for e in frame_ids]
    start_time = frame_time[0]
    frame_time = ", ".join(["{:.2f}s".format(time - start_time) for time in frame_time])
    time_meta['frame_time'] = frame_time
    return buffer, time_meta





def hand_obj_ann_loader(root, handobj_root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    
    assert fps > 0, 'fps should be greater than 0' 
    time_meta = {}
    import matplotlib.pyplot as plt
    time_meta['duration'] = end_second - second

    assert end_second > second, 'end_second should be greater than second'

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
        handobj_df = pd.read_csv(osp.join(handobj_root, '{}.{}'.format(vid, ext), '{}.{}.csv'.format(chunk, ext)))
        hand_dets_list = handobj_df.iloc[rel_frame_ids]['hand_dets'].tolist()
        obj_dets_list = handobj_df.iloc[rel_frame_ids]['obj_dets'].tolist()

        try:
            frames = vr.get_batch(rel_frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
        except IndexError:
            print('IndexError', root, vid, ext, second, end_second)

    for i in range(frames.shape[0]):

        hand_dets_list[i] = np.array(ast.literal_eval(hand_dets_list[i])) if hand_dets_list[i] != '[]' else np.nan
        obj_dets_list[i] = np.array(ast.literal_eval(obj_dets_list[i])) if obj_dets_list[i] != '[]' else np.nan

    return frames, hand_dets_list, obj_dets_list    

def avion_video_render_loader(root, handobj_root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    assert fps > 0, 'fps should be greater than 0' 
    time_meta = {}
    import matplotlib.pyplot as plt
    time_meta['duration'] = end_second - second

    assert end_second > second, 'end_second should be greater than second'

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
        handobj_df = pd.read_csv(osp.join(handobj_root, '{}.{}'.format(vid, ext), '{}.{}.csv'.format(chunk, ext)))
        hand_dets_list = handobj_df.iloc[rel_frame_ids]['hand_dets'].tolist()
        obj_dets_list = handobj_df.iloc[rel_frame_ids]['obj_dets'].tolist()

        try:
            frames = vr.get_batch(rel_frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
        except IndexError:
            print('IndexError', root, vid, ext, second, end_second)

        if frames.shape[0] == 0:
            continue
        
        # aa = 1
        # show one of the frames
        # plt.figure()
        # plt.imshow(frames[0])
        # plt.savefig('frame.png')
        # plt.close()
        
        frames = render_frames(frames, hand_dets_list, obj_dets_list, thresh_hand=0.5, thresh_obj=0.5)

        plt.figure()
        plt.imshow(frames[0])
        plt.savefig('frame_rendered.png')
        plt.close()

        all_frames.append(frames)
        all_frame_ids.append(frame_ids)
        if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
            break
    res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
    time_meta['n_frames'] = res.shape[0]
    all_frame_ids = np.concatenate(all_frame_ids, axis = 0)
    frame_time = [e/fps for e in all_frame_ids]
    frame_time-= frame_time[0]
    frame_time = ", ".join([f"{i:.2f}s" for i in frame_time])
    time_meta['frame_time'] = frame_time
    assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
    return res, time_meta

def render_frames(frames, hand_dets_list, obj_dets_list, thresh_hand=0.5, thresh_obj=0.5):
    """
    Render frames with hand and object detections
    """
    rendered_frames = []
    for i in range(frames.shape[0]):
        rendered_frame = render_frame(frames[i], hand_dets_list[i], obj_dets_list[i], thresh_hand, thresh_obj)
        rendered_frames.append(rendered_frame)
    return np.array(rendered_frames)




if __name__ == '__main__':

    anno_root = "/storage-rcp-pure/upmwmathis_scratch/shaokai/epic-kitchens-100-annotations/"
    #generator = MultiChoiceGenerator(anno_root)
    generator = AvionMultiChoiceGenerator(anno_root)
    import json

    with open('/storage-rcp-pure/upmwmathis_scratch/shaokai/avion_predictions_train.json') as f:
        predictions = json.load(f)

    print (len(predictions))
    print (predictions['0'])
    print (len(predictions['0']['predictions']))
    

    pass