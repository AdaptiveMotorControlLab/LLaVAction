import csv 
import numpy as np
import random
import os
import decord
import os.path as osp
import torch

def generate_label_map(anno_root):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    # from id to name
    verb_maps = {}
    noun_maps = {}
    for f in [      
        os.path.join(anno_root,'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip the header
        for row in csv_reader:
            
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            narration = row[8]
            if row[10] not in verb_maps.keys():
                verb_maps[row[10]] = row[9]
            if row[12] not in noun_maps.keys():
                noun_maps[row[12]] = row[11]

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
    return labels, mapping_vn2act, verb_maps, noun_maps

def generate_unique_label_map(anno_root):
    """
    The problem with generate_label_map is that if a noun class or a verb class is already mapped
    to a specific narration at that instance, the subsequent noun class and verb class will continue to use that previous mapping,

    """
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    # from id to name
    train_verb_maps = {}
    train_noun_maps = {}
    val_verb_maps = {}
    val_noun_maps = {}
    
    for f in [      
        os.path.join(anno_root,'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip the header

        verb_maps = train_verb_maps if 'train.csv' in f else val_verb_maps
        noun_maps = train_noun_maps if 'train.csv' in f else val_noun_maps

        for idx, row in enumerate(csv_reader):
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            narration = row[8]
            verb_maps[idx] = row[9]
            noun_maps[idx] = row[11]

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
    return labels, mapping_vn2act, train_verb_maps, train_noun_maps, val_verb_maps, val_noun_maps


def format_task_related_prompt(option_list):
    prefix = "The video is taken from egocentric view. What action is the person performing? Given multiple choices, format your answer as the 'option letter. option_name' such as 'A. move knife' where A is the option letter and knife is the option_name.\n"
    assert isinstance(option_list, list)
    suffix = ",".join(option_list)
    suffix = "Here are the options you are tasked:\n" + suffix 
    ret = prefix + suffix
    return ret

def format_time_instruction(video_duration, n_frames, include_frame_time = False):

    prefix = f"You are seeing a video taken from egocentric view. The video lasts for {video_duration:.2f} seconds, and {n_frames} frames are uniformly sampled from it."

    frame_time = [i * (video_duration / n_frames) for i in range(n_frames)]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    
    suffix = ""
    if include_frame_time:
        suffix = f"These frames are located at {frame_time}."
    
    return prefix + suffix


def format_llava_prompt(image_token, 
                        option_list, 
                        video_duration,
                        n_frames,
                        include_time_instruction = False,
                        include_frame_time = False
                        ):
    """
    baseline llava prompt: {image_token}\n{task_related_prompt}
    with time instruction: {image_token}\n{time_instruction}\n{task_related_prompt}

    """
    task_related_prompt = format_task_related_prompt(option_list)
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

class MultiChoiceGenerator:
    """
    Generating multi choice
    """
    def __init__(self, ann_root):
        self.ann_root = ann_root
        _, self.mapping_vn2act, self.verb_maps, self.noun_maps = generate_label_map(ann_root)
    

    def generate_multi_choice(self, gt_vn, k):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list

        """        

        # let v_id and n_id be string type
        gt_v_id, gt_n_id = gt_vn.split(':')    
        assert isinstance(gt_v_id, str) and isinstance(gt_n_id, str)
        gt_v_name, gt_n_name = self.verb_maps[gt_v_id], self.noun_maps[gt_n_id]

        # letters as A, B, C, D, .. Note we maximally support 26 letters
        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]
        vn_list = list(self.mapping_vn2act.keys())
        action_list = [f"{self.verb_maps[e.split(':')[0]]} {self.noun_maps[e.split(':')[1]]}" for e in vn_list]
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
                'valid_letters': letters
            }
        
        return data
    
class AvionMultiChoiceGenerator(MultiChoiceGenerator):
    """
    Generate multichoice using avion predictions
    """
    def __init__(self, ann_root):
        super().__init__(ann_root)
    
    def generate_multi_choice(self, gt_vn, avion_predictions, k):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list that is not gt_vn (this is important as avion_predictions can contain correct prediction)        

        """    
        gt_v_id, gt_n_id = gt_vn.split(':')
        gt_v_name, gt_n_name = self.verb_maps[gt_v_id], self.noun_maps[gt_n_id]
        gt_answer = f'{gt_v_name} {gt_n_name}'

        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]

        # we should have plenty of predictions to select, so let's not always pick the hardest
        assert len(avion_predictions) > 2*k
        avion_predictions = avion_predictions[:k*2]
        avion_predictions = parse_avion_predictions(avion_predictions)
        if gt_answer in avion_predictions:
            avion_predictions.remove(gt_answer)
        # just so that it's not strictly desending with confidence
        random.shuffle(avion_predictions)
        avion_predictions = avion_predictions[:k-1]

        answers = [gt_answer] + avion_predictions
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
                'valid_letters': letters
            }        
        return data
    
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

def create_multi_choice_from_avion_predictions(avion_predictions, k)-> dict:
    
    letters = [chr(65+i) for i in range(26)][:k]
    options = list(range(26))[:k]

    predictions = avion_predictions[:k]
    predictions = parse_avion_predictions(predictions)    

    for i in range(len(options)):              
        options[i] = f'{letters[i]}. {predictions[i]}'
                
    mc_data = {
        'options': {0: options},
        'valid_letters': letters,
        'avion_pred': predictions[0]
        }    
    
    return mc_data
    


def avion_video_loader(root, vid, ext, second, end_second,
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
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        time_meta['n_frames'] = res.shape[0]
        all_frame_ids = np.concatenate(all_frame_ids, axis = 0)
        frame_time = [e/fps for e in all_frame_ids]
        frame_time-= frame_time[0]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        time_meta['frame_time'] = frame_time
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res, time_meta


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
    
    print (generator.generate_multi_choice('3:3',  predictions['0']['predictions'],  5))

    pass