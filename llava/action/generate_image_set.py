import numpy as np
import pandas as pd
import os
import csv
import decord
import torch
import ast
import copy
from tqdm import tqdm
import json
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from render_utils import render_frame
from auxfun_videos import VideoReader

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

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def avion_video_handobj_loader(root, handobj_root, vid, ext, second, end_second, origin_fps,
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
        video_filename = os.path.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
        if not os.path.exists(video_filename):
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
    # only leave the unique frame_ids
    frame_ids = list(set(frame_ids))

    video_path_list = []
    all_rel_frame_ids = []
    left_hand_states = []
    right_hand_states = []
    debug_hand_list = []
    debug_obj_list = []
    # allocate absolute frame-ids into the relative ones
    for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
        rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
        rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
        if len(rel_frame_ids) == 0:
            continue
        # vr = get_video_reader(
        #     os.path.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
        #     num_threads=threads,
        #     fast_rrc=fast_rrc, rrc_params=rrc_params,
        #     fast_rcc=fast_rcc, rcc_params=rcc_params,
        # )

        video_path_list.extend([os.path.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext))] * len(rel_frame_ids))
        all_rel_frame_ids.extend(rel_frame_ids)


        handobj_df = pd.read_csv(os.path.join(handobj_root, '{}.{}'.format(vid, ext), '{}.{}.csv'.format(chunk, ext)))
        hand_dets_list = handobj_df.iloc[rel_frame_ids]['hand_dets'].tolist()
        obj_dets_list = handobj_df.iloc[rel_frame_ids]['obj_dets'].tolist()

        debug_hand_list.extend(copy.deepcopy(hand_dets_list))
        debug_obj_list.extend(copy.deepcopy(obj_dets_list))

        for i, hand_dets in enumerate(hand_dets_list):
            hand_dets = np.array(ast.literal_eval(hand_dets)) if hand_dets != '[]' else None
            if hand_dets is not None:
                left_hand_dets = hand_dets[np.logical_and(hand_dets[:, -1] == 0, hand_dets[:, 4] > 0.5)]
                right_hand_dets = hand_dets[np.logical_and(hand_dets[:, -1] == 1, hand_dets[:, 4] > 0.5)]
                if len(left_hand_dets) > 0:
                    left_hand_det = left_hand_dets[np.argmax(left_hand_dets[:, 4])]
                    left_hand_states.append(int(left_hand_det[5]))
                else:
                    left_hand_states.append(-1)
                if len(right_hand_dets) > 0:
                    right_hand_det = right_hand_dets[np.argmax(right_hand_dets[:, 4])]
                    right_hand_states.append(int(right_hand_det[5]))
                else:
                    right_hand_states.append(-1)
            else:
                left_hand_states.append(-1)
                right_hand_states.append(-1)

        # try:
        #     frames = vr.get_batch(rel_frame_ids).asnumpy()
        # except IndexError:
        #     print('IndexError', root, vid, ext, second, end_second)
        # if frames.shape[0] == 0:
        #     continue
        # all_frames.append(frames)

    assert len(left_hand_states) == len(right_hand_states) == len(frame_ids) == len(debug_hand_list) == len(debug_obj_list)
    # obtain the original frame id based on the original fps and the current fps
    origin_frame_ids = [int(frame_id * origin_fps / fps) for frame_id in frame_ids]
    return origin_frame_ids, left_hand_states, right_hand_states, debug_hand_list, debug_obj_list, video_path_list, all_rel_frame_ids

def save_video_data(json_data, frame_root, save_root, image_weight, image_height, original_image_dir, resized_image_dir):
    idx = json_data['id']
    vid_path = json_data['vid_path'].split('/')[-1]
    original_frame_id = json_data['frame_id']
    rel_frame_id = json_data['rel_frame_id']

    # read the image
    frame_path = os.path.join(frame_root, vid_path, 'frame_{:0>10d}.jpg'.format(original_frame_id))
    # copy the original image
    shutil.copy(frame_path, os.path.join(save_root, original_image_dir, '{:0>8d}.jpg'.format(idx)))

    # # check if the obtained data are correct
    # original_image = Image.open(frame_path)
    # original_image = original_image.resize((image_weight, image_height))
    # rendered_original_images = render_frame(np.array(original_image), json_data['debug_hand_list'], json_data['debug_obj_list'], thresh_hand=0.5, thresh_obj=0.5)
    # # change to PIL image
    # rendered_original_images = Image.fromarray(rendered_original_images)
    # rendered_original_images.save('rendered_original_images.png')

    video_path = json_data['video_path_list']
    video_reader = VideoReader(video_path)
    video_reader.set_to_frame(rel_frame_id)
    resized_image = video_reader.read_frame()
    # save the image
    resized_image = Image.fromarray(resized_image)
    resized_image.save(os.path.join(save_root, resized_image_dir, '{:0>8d}.jpg'.format(idx)))

    # # check if the obtained data are correct
    # rendered_resized_image = render_frame(np.array(resized_image), json_data['debug_hand_list'], json_data['debug_obj_list'], thresh_hand=0.5, thresh_obj=0.5)
    # # change to PIL image
    # rendered_resized_image = Image.fromarray(rendered_resized_image)
    # rendered_resized_image.save('rendered_resized_image.png')

def test_json_data(json_data, save_root, image_weight, image_height, original_image_dir, resized_image_dir):
    idx = json_data['id']
    vid_path = json_data['vid_path'].split('/')[-1]
    original_frame_id = json_data['frame_id']
    rel_frame_id = json_data['rel_frame_id']

    # read the image
    original_frame_path = os.path.join(save_root, original_image_dir, '{:0>8d}.jpg'.format(idx))

    # check if the obtained data are correct
    original_image = Image.open(original_frame_path)
    original_image = original_image.resize((image_weight, image_height))
    rendered_original_images = render_frame(np.array(original_image), json_data['debug_hand_list'], json_data['debug_obj_list'], thresh_hand=0.5, thresh_obj=0.5)
    # change to PIL image
    rendered_original_images = Image.fromarray(rendered_original_images)
    rendered_original_images.save('rendered_original_images.png')

    resized_frame_path = os.path.join(save_root, resized_image_dir, '{:0>8d}.jpg'.format(idx))
    resized_image = Image.open(resized_frame_path)

    # check if the obtained data are correct
    rendered_resized_image = render_frame(np.array(resized_image), json_data['debug_hand_list'], json_data['debug_obj_list'], thresh_hand=0.5, thresh_obj=0.5)
    # change to PIL image
    rendered_resized_image = Image.fromarray(rendered_resized_image)
    rendered_resized_image.save('rendered_resized_image.png')

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

if __name__ == '__main__':
    # annotation_root = '/mediaPFM/data/haozhe/EK100/epic-kitchens-100-annotations'
    # video_root = '/mediaPFM/data/haozhe/onevision/llava_video/EK100'
    # handobj_root = '/mediaPFM/data/haozhe/EK100/handobj_results'
    # save_root = '/mediaPFM/data/haozhe/EK100/handobj_imageset'
    # frame_root = '/mnt/haozhe_dexycb/all_rgb_frames'

    annotation_root = '/media/data2/Kitchen_data/smart_kitchen/epic_kitchen/epic-kitchens-100-annotations'
    video_root = '/media/data1/Kitchen/avion/datasets/EK100/EK100_320p_15sec_30fps_libx264'
    handobj_root = 'Save_dir'
    save_root = '/media/data2/Kitchen_data/smart_kitchen/epic_kitchen/haozhe/handobj_imageset'
    frame_root = '/media/data2/Kitchen_data/smart_kitchen/epic_kitchen/EPIC-KITCHENS/all_rgb_frames'

    
    split = 'train' # train, validation
    clip_length = 8
    image_weight = 568
    image_height = 320
    resized_image_dir = 'resized_images'
    original_image_dir = 'original_images'

    sample_nums = 10000 if split == 'validation' else 50000

    hand_types = [-1, 0, 1, 3, 4]

    save_root = os.path.join(save_root, split)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, resized_image_dir), exist_ok=True)
    os.makedirs(os.path.join(save_root, original_image_dir), exist_ok=True)
    annotation_file = os.path.join(annotation_root, 'EPIC_100_{}.csv'.format(split))

    data_dict = {'vid_path': [], 'gt_vn': [], 'narration': [], 'frame_id': [], 'left_hand_state': [], 'right_hand_state': [], 'debug_hand_list': [], 'debug_obj_list': [], 'video_path_list': [], 'rel_frame_id': []}

    csv_reader = csv.reader(open(annotation_file))
    possible_fps = np.array([60, 50, 30])
    _ = next(csv_reader) # skip the header
    for row in tqdm(csv_reader):
        narration = row[8]
        pid, vid = row[1:3]
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_frame, end_frame = int(row[6]), int(row[7])
        # origin_fps = (end_frame - start_frame) / (end_second - start_second)
        origin_fps = end_frame / end_second
        fps_diff = np.min(np.abs(possible_fps - origin_fps))
        if fps_diff > 2:
            print('fps_diff is too large, the computed fps is {}'.format(origin_fps))
        else:
            origin_fps = possible_fps[np.argmin(np.abs(possible_fps - origin_fps))]
        vid_path = '{}/{}'.format(pid, vid)
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun)
        narration = row[8]

        frame_ids, left_hand_states, right_hand_states, debug_hand_list, debug_obj_list, video_path_list, all_rel_frame_ids = \
            avion_video_handobj_loader(video_root, handobj_root, vid_path, 'MP4', start_second, end_second, origin_fps,
                                   chunk_len = 15, clip_length = clip_length, threads = 1, fast_rrc=False, fast_rcc = False, jitter = False)
        

        # add to data_dict
        data_dict['vid_path'].extend([vid_path] * len(frame_ids))
        data_dict['gt_vn'].extend([gt_vn] * len(frame_ids))
        data_dict['narration'].extend([narration] * len(frame_ids))
        data_dict['frame_id'].extend(frame_ids)
        data_dict['left_hand_state'].extend(left_hand_states)
        data_dict['right_hand_state'].extend(right_hand_states)
        data_dict['debug_hand_list'].extend(debug_hand_list)
        data_dict['debug_obj_list'].extend(debug_obj_list)
        data_dict['video_path_list'].extend(video_path_list)
        data_dict['rel_frame_id'].extend(all_rel_frame_ids)

    total_samples = len(data_dict['frame_id'])
    # change to numpy array
    for key in data_dict.keys():
        if key in ['debug_hand_list', 'debug_obj_list', 'video_path_list']:
            continue
        data_dict[key] = np.array(data_dict[key])
        assert len(data_dict[key]) == total_samples


    type_id_array = np.zeros((total_samples, len(hand_types))).astype(np.bool_)
    selected_ids = np.zeros(total_samples).astype(np.bool_)
    left_types = []
    for i, hand_type in enumerate(hand_types):
        type_id_array[:, i] = np.logical_or(data_dict['left_hand_state'] == hand_type, data_dict['right_hand_state'] == hand_type)
        print('hand type {} has {} samples'.format(hand_type, np.sum(type_id_array[:, i])))
        if np.sum(type_id_array[:, i]) < int(sample_nums * 0.25):
            selected_ids = np.logical_or(selected_ids, type_id_array[:, i])
        else:
            left_types.append([i, np.sum(type_id_array[:, i])])

    left_per_nums = (sample_nums - np.sum(selected_ids)) // len(left_types)
    # start from type with the least number of samples
    left_types = sorted(left_types, key=lambda x: x[1])
    for i, (type_id, type_nums) in enumerate(left_types):
        # select samples that are not selected yet
        type_ids = np.logical_and(type_id_array[:, type_id], np.logical_not(selected_ids))
        type_ids_index = np.random.choice(np.where(type_ids)[0], left_per_nums, replace=False)
        type_ids = np.zeros(total_samples).astype(np.bool_)
        type_ids[type_ids_index] = True
        selected_ids = np.logical_or(selected_ids, type_ids)

    seletect_indexes = np.where(selected_ids)[0]
    for key in data_dict.keys():
        if key in ['debug_hand_list', 'debug_obj_list', 'video_path_list']:
            data_dict[key] = [data_dict[key][i] for i in seletect_indexes]
        else:
            data_dict[key] = data_dict[key][selected_ids]

    # check the number of samples for each hand type
    for i, hand_type in enumerate(hand_types):
        type_id = np.logical_or(data_dict['left_hand_state'] == hand_type, data_dict['right_hand_state'] == hand_type)
        print('After selection, hand type {} has {} samples'.format(hand_type, np.sum(type_id)))
        
    total_samples = len(data_dict['frame_id'])
    # change the keys in data dict back to list
    for key in data_dict.keys():
        if isinstance(data_dict[key], np.ndarray):
            data_dict[key] = data_dict[key].tolist()

    # save the data_dict as jsonl file
    jsonl_data = []
    for sample_id in range(total_samples):
        json_data = {}
        json_data['id'] = sample_id
        for key in data_dict.keys():
            json_data[key] = data_dict[key][sample_id]
        jsonl_data.append(json_data)

    jsonl_file = os.path.join(save_root, 'EPIC_100_handobj_imageset_{}_{}.jsonl'.format(split, clip_length))
    with open(jsonl_file, 'w') as f:
        for json_data in jsonl_data:
            f.write(json.dumps(json_data) + '\n')

    # # save video data
    # for json_data in jsonl_data:
    #     save_video_data(json_data, frame_root, save_root, image_weight, image_height, original_image_dir, resized_image_dir)

    # save the video in parallel
    Parallel(n_jobs=6)(delayed(save_video_data)(json_data, frame_root, save_root, image_weight, image_height, original_image_dir, resized_image_dir) for json_data in tqdm(jsonl_data))

