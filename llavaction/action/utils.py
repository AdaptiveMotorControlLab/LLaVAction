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
from llavaction.action.render_utils import render_frame
from collections import defaultdict
import json
from llavaction.utils import rank0_print
import re
# set random seed
random.seed(42)



def remove_sub_nouns(nlp, narration, verb, nouns):
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
        doc = nlp(narration)       
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
         
    else:
        new_narration = narration
        
    return new_narration


def remove_option_letter(answer):
    if '. ' in answer:
        return answer.split('. ')[1]
    else:
        return answer

def generate_label_map(anno_root, action_representation):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    mapping_vnstr2narration = defaultdict(list)
    
    # Load CSVs
    noun_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_noun_classes_v2.csv'))
    verb_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_verb_classes.csv'))
    
    # Initialize maps
    verb_maps = {} if 'key' in action_representation or 'gpt_narration' in action_representation or action_representation == 'first_sample' else None
    noun_maps = {} if 'key' in action_representation or 'gpt_narration' in action_representation or action_representation == 'first_sample' else None
    
    # Process verb and noun maps
    if 'key' in action_representation or 'gpt_narration' in action_representation:
        for _, row in verb_classes_pd.iterrows():
            verb_maps[str(row['id'])] = row['key']
        for _, row in noun_classes_pd.iterrows():
            elements = row['key'].split(':')
            noun_maps[str(row['id'])] = ' '.join(elements[1:] + [elements[0]]) if len(elements) > 1 else row['key']   
    # Batch processing setup
    if 'cut' in action_representation:
        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
        
        def process_batch_of_rows(rows_batch):
            # Prepare data for batch processing
            narrations = []
            verbs = []
            nouns = []
            vns = []
            
            for row in rows_batch:
                narrations.append(row[8])
                verbs.append(row[9])
                nouns.append(row[13])
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                vns.append(vn)
            
            # Process all narrations in batch
            processed_narrations = []
            for doc, verb, noun in zip(nlp.pipe(narrations, batch_size=1000), verbs, nouns):
                processed_narration = remove_sub_nouns_with_doc(doc, verb, noun)
                processed_narrations.append(processed_narration)
            
            return zip(vns, processed_narrations)

    # Process files
    batch_size = 1000
    current_batch = []
    
    for f in [
        os.path.join(anno_root, 'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        next(csv_reader)  # skip header
        
        for row in tqdm(csv_reader):
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            if vn not in vn_list:
                vn_list.append(vn)
                
            if action_representation == 'first_sample':
                if row[10] not in verb_maps:
                    verb_maps[row[10]] = row[9]
                if row[12] not in noun_maps:
                    noun_maps[row[12]] = row[11]
            
            if 'cut' in action_representation:
                current_batch.append(row)
                
                if len(current_batch) >= batch_size:
                    # Process batch
                    for batch_vn, processed_narration in process_batch_of_rows(current_batch):
                        if batch_vn not in mapping_vn2narration:
                            mapping_vn2narration[batch_vn] = [processed_narration]
                        else:
                            mapping_vn2narration[batch_vn].append(processed_narration)
                    current_batch = []
            else:
                narration = row[8]
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
            if verb_maps and noun_maps:
                temp_v, temp_n = vn.split(':')
                temp_v, temp_n = verb_maps[temp_v], noun_maps[temp_n]
                mapping_vnstr2narration[f'{temp_v} {temp_n}'].append(narration)
        # Process remaining batch
        if current_batch and 'cut' in action_representation:
            for batch_vn, processed_narration in process_batch_of_rows(current_batch):
                if batch_vn not in mapping_vn2narration:
                    mapping_vn2narration[batch_vn] = [processed_narration]
                else:
                    mapping_vn2narration[batch_vn].append(processed_narration)
    
    # Finalize results
    vn_list = sorted(vn_list)
    print('# of action= {}'.format(len(vn_list)))
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
    
    # Create labels with frequency sorting
    labels = {}
    for vn, narrations in mapping_vn2narration.items():
        frequency_count = Counter(narrations)
        sorted_unique_list = [item for item, count in frequency_count.most_common()]
        labels[vn] = sorted_unique_list

    return labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps

def remove_sub_nouns_with_doc(doc, verb: str, noun: str) -> str:
    """Process a spaCy doc instead of raw text for noun substitution"""
    # Your existing noun substitution logic here, but working with doc instead of text
    # Modify your existing remove_sub_nouns function to work with a doc directly
    processed_text = doc.text  # Replace with your actual processing logic
    return processed_text


def format_task_related_prompt(question, question_type, meta_data = None, perspective = "first_person", learn_neighbor_actions = ""):
    """
    Task related prompt is impacted by the question_type.
    We currently support mc_{action_representation} and gpt-gt-reason
    We are thinking about tweaking the prompt based on the action representation.
    """
    
    if perspective == "first_person":
        perspective_prefix = "You are seeing this video from egocentric view and you are the person. Your hands are sometimes interacting with objects. What action are you doing? "
    elif perspective == "third_person":
        perspective_prefix = "The video is taken from egocentric view. The person's hands are sometimes interacting with objects. What action is the person doing?"
    
    if learn_neighbor_actions == "prior" and meta_data:
        prev2_narration = meta_data['prev2_narration']
        prev2_offset = meta_data['prev2_offset']
        prev1_narration = meta_data['prev1_narration']
        prev1_offset = meta_data['prev1_offset']
        cur_narration = meta_data['cur_narration']
                   
    if question_type.startswith("mc_") or question_type.startswith('temporal_cot'):
                                      
        if question_type.startswith("mc_") and learn_neighbor_actions == "prior" and meta_data and random.random() < 0.3:
            # this means it's training time and we are learning the prior actions
            prefix = f"{perspective_prefix}\n"
            assert isinstance(question, list)
            suffix = ", ".join(question)
            
            suffix = f"{prev2_offset} seconds ago, you started an action {prev2_narration}. {prev1_offset} seconds ago, you started an action {prev1_narration}. What action are you currently performing? Here are the options of actions you can select:\n" + suffix 
            ret = prefix + suffix
        elif question_type.startswith("temporal_cot") and learn_neighbor_actions == "prior" and meta_data:
            # means it's test time
            if question_type == 'temporal_cot_caption':
                ret = f"{perspective_prefix} {prev2_offset} seconds ago, you started an action {prev2_narration}. {prev1_offset} seconds ago, you started an action {prev1_narration}. Describe in details what you see from the video frames. You must talk in the first person perspective. Try to focus on what you are doing."
                rank0_print(ret)
            else:
                prefix = f"{perspective_prefix}\n"
                assert isinstance(question, list)
                suffix = ", ".join(question)           
                suffix = f"{prev2_offset} seconds ago, you started an action {prev2_narration}. {prev1_offset} seconds ago, you started an action {prev1_narration}. What action are you currently performing? Here are the options of actions you can select:\n" + suffix 
                ret = prefix + suffix             
            
        else:
            action_rep_suffix = "Given multiple choices, format your answer briefly such as 'A. move knife'. "              
            prefix = f"{perspective_prefix}{action_rep_suffix}\n"
            assert isinstance(question, list)
            suffix = ", ".join(question)
            suffix = "Here are the options of actions you are selecting:\n" + suffix 
            ret = prefix + suffix
    
    elif question_type == "direct_narration":                  
        ret = f"{perspective_prefix} What action are you performing? Give a short sentence such as 'move knife'."
            
    elif question_type == "temporal_detection":
        ret = question
    elif question_type == "gpt-gt-reason" or question_type == "caption":
        ret = f"{perspective_prefix} Describe in details what you see from the video frames. You must talk in the first person perspective. Try to focus on what you are doing. "
           
    
    elif question_type == "gpt-gt-strong-reason":
        ret = f"{perspective_prefix} Describe in details what you see and answer the multi-choice question. Explain why wrong answers are wrong and why the correct answer is correct. "
        suffix = ", ".join(question)
        suffix = "Here are the options of actions you are selecting:\n" + suffix  
        ret = ret + suffix       

    elif question_type == "dpo":
        ret = "You are seeing this video from egocentric view and you are the person. Your hands are sometimes interacting with obects. Describe in details what you see and what you are doing."

    elif question_type == "open-ended":
        ret = f"You are seeing this video from egocentric view and you are the person. {question}"

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
                        include_frame_time = False,
                        meta_data = None,
                        learn_neighbor_actions = "",
                        perspective = "first_person"
                        ):
    """
    baseline llava prompt: {image_token}\n{task_related_prompt}
    with time instruction: {image_token}\n{time_instruction}\n{task_related_prompt}
    """
    task_related_prompt = format_task_related_prompt(question, question_type, 
                                                     meta_data = meta_data, 
                                                     learn_neighbor_actions = learn_neighbor_actions,
                                                     perspective = perspective)

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


def get_gpt_narration_from_vn_str(vn_str):
    with open('mapping_vnstr2narration_gpt.json') as f:
        data_dict = json.load(f)        
    return data_dict[vn_str]

def parse_vn_ids(answer_id, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps):
    answer_items = []
    if 'key' in action_representation or action_representation == 'first_sample':
        v_id, n_id = answer_id.split(':')
        v_name, n_name = verb_maps[v_id], noun_maps[n_id]
        answer_items.append(f'{v_name} {n_name}')
    if 'gpt_narration' in action_representation:        
        v_id, n_id = answer_id.split(':')
        v_name, n_name = verb_maps[v_id], noun_maps[n_id]
        vn_str = f'{v_name} {n_name}'
        gpt_narration = get_gpt_narration_from_vn_str(vn_str)
        answer_items.append(gpt_narration['new_gt'])         
             
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

class MultiChoiceGenerator:
    """
    Generating multi choice
    """
    def __init__(self, ann_root):
        self.ann_root = ann_root
        
class RandomMultiChoiceGenerator(MultiChoiceGenerator):
    def __init__(self, ann_root):
        super().__init__(ann_root)
    
    def generate_multi_choice(self, 
                              gt_vn, 
                              narration, 
                              k, 
                              action_representation, 
                              n_narrations, 
                              labels, 
                              mapping_vn2narration, 
                              verb_maps, 
                              noun_maps,
                              is_train = True,
                              benchmark_testing = False
                              ):

        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list

        """        
        if is_train:
            return self.train_generate(gt_vn, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
        else:
            return self.test_generate(gt_vn, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps, benchmark_testing = benchmark_testing)
    
    def train_generate(self, gt_vn, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps, benchmark_testing = False):
        # letters as A, B, C, D, .. Note we maximally support 26 letters
        letters = [chr(65+i) for i in range(26)][:k]                
        
        answer_list = [vn for vn in mapping_vn2narration.keys()]                                
        wrong_answers = np.random.choice(answer_list, size = k-1, replace = False)        
        answer_ids = [gt_vn] + list(wrong_answers)
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
    
    def test_generate(self, gt_vn, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps, benchmark_testing = False):
        """
        There is no difference between train and test for random generation
        """        
        return self.train_generate(gt_vn, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps, benchmark_testing = benchmark_testing)        

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

    def test_generate(self, 
                      gt_vn, 
                      action_model_predictions, 
                      narration, 
                      k, 
                      action_representation, 
                      n_narrations, 
                      labels, 
                      mapping_vn2narration, 
                      verb_maps, 
                      noun_maps,
                      benchmark_testing = False
                      ):
        """
        During testing, we use the top k predictions from avion. No randomness. We do not mix the gt_vn with the avion predictions
        """        
        answer_ids = action_model_predictions[:k]
        
        if benchmark_testing:
            # if we are testing on benchmark, we need to ensure that the gt_vn is in the top k predictions
            # if not, we remove the last prediction and add the gt_vn
            if gt_vn not in answer_ids:
                answer_ids.pop()
                answer_ids.append(gt_vn)
        else:
           pass
                      
        answers = []
        for answer_id in answer_ids:
            answer = parse_vn_ids(answer_id, gt_vn, narration, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
            answers.append(answer)
        avion_pred = answers[0]
        
        random.shuffle(answers)
        
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
                'avion_pred': avion_pred,
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
                              is_train = True,
                              benchmark_testing = False
                              ):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list that is not gt_vn (this is important as avion_predictions can contain correct prediction)        

        """    
        if is_train:
            return self.train_generate(gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps)
        else:
            return self.test_generate(gt_vn, avion_predictions, narration, k, action_representation, n_narrations, labels, mapping_vn2narration, verb_maps, noun_maps, benchmark_testing = benchmark_testing)
    
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

    ann_root = '/data/EK100/epic-kitchens-100-annotations'
    #    action_representation = 'official_key' # gpt_narration
    action_representation = 'gpt_narration'
    
    mc = AvionMultiChoiceGenerator(ann_root)
    
    #generate_label_map(ann_root, action_representation)
    
    mc.test_generate()