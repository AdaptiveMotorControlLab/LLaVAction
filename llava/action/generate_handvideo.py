import numpy as np
import os
import pandas as pd
import decord
import ast
import cv2

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

def crop_hands(video_reader, sample_frames, df, frame_id):
    image_size = (384, 384)
    # video_size = (1920, 1080)
    handobj_size = (568, 320)
    expand_ratio = 1.5
    minimum_size = 20

    # get the frame
    try:
        frame = video_reader[sample_frames[frame_id]].asnumpy()
    except:
        hand_image = np.zeros((image_size[0], image_size[1]*2, 3), dtype=np.uint8)
        return hand_image

    video_size = (frame.shape[1], frame.shape[0])
    # get the hand detection results
    hand_dets = df.iloc[frame_id]['hand_dets']

    # change the string to list
    hand_dets = np.array(ast.literal_eval(hand_dets)) if hand_dets != '[]' else None

    left_image = np.zeros(image_size + (3,), dtype=np.uint8)
    right_image = np.zeros(image_size + (3,), dtype=np.uint8)

    if hand_dets is not None:
        # select the left hand detection with the highest score
        left_hand = hand_dets[hand_dets[:, -1] == 0]
        if len(left_hand) > 0:
            left_hand = left_hand[np.argmax(left_hand[:, 4])]
            bbox = [left_hand[0] * video_size[0] / handobj_size[0], left_hand[1] * video_size[1] / handobj_size[1],
                    left_hand[2] * video_size[0] / handobj_size[0], left_hand[3] * video_size[1] / handobj_size[1]]
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 > minimum_size:
                # expand the bbox based on the expand_ratio and the longer side, and make the bbox square
                half_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                bbox = [center[0] - half_side * expand_ratio, center[1] - half_side * expand_ratio,
                        center[0] + half_side * expand_ratio, center[1] + half_side * expand_ratio]
                bbox = [int(np.round(x)) for x in bbox]

                # crop the image with the bbox and zero padding
                cropped_image = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8)
                cropped_image[max(0, -bbox[1]):min(bbox[3] - bbox[1], video_size[1] - bbox[1]),
                                max(0, -bbox[0]):min(bbox[2] - bbox[0], video_size[0] - bbox[0]), :] = frame[max(bbox[1], 0):min(bbox[3], video_size[1]), max(bbox[0], 0):min(bbox[2], video_size[0])]
                
                # resize the cropped image to the image_size
                left_image = cv2.resize(cropped_image, image_size)

        
        # select the right hand detection with the highest score
        right_hand = hand_dets[hand_dets[:, -1] == 1]
        if len(right_hand) > 0:
            right_hand = right_hand[np.argmax(right_hand[:, 4])]
            bbox = [right_hand[0] * video_size[0] / handobj_size[0], right_hand[1] * video_size[1] / handobj_size[1],
                    right_hand[2] * video_size[0] / handobj_size[0], right_hand[3] * video_size[1] / handobj_size[1]]
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 > minimum_size:
                # expand the bbox based on the expand_ratio and the longer side, and make the bbox square
                half_side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                bbox = [center[0] - half_side * expand_ratio, center[1] - half_side * expand_ratio,
                        center[0] + half_side * expand_ratio, center[1] + half_side * expand_ratio]
                bbox = [int(np.round(x)) for x in bbox]

                # crop the image with the bbox and zero padding
                cropped_image = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8)
                cropped_image[max(0, -bbox[1]):min(bbox[3] - bbox[1], video_size[1] - bbox[1]), 
                                max(0, -bbox[0]):min(bbox[2] - bbox[0], video_size[0] - bbox[0]), :] = frame[max(bbox[1], 0):min(bbox[3], video_size[1]), max(bbox[0], 0):min(bbox[2], video_size[0])]

                # resize the cropped image to the image_size
                right_image = cv2.resize(cropped_image, image_size)

    # concatenate the left and right hand images
    hand_image = np.concatenate((left_image, right_image), axis=1)

    return hand_image[:, :, ::-1]

def process_clip(clips, video_path, handobj_path, save_video_path, clip_i):
    seconds = 15
    handobj_fps = 30
    image_size = (384, 384)
    video_reader = decord.VideoReader(video_path)
    video_fps = video_reader.get_avg_fps()
    

    clip = clips[clip_i]
    clip_path = os.path.join(handobj_path, clip)
    save_clip_path = os.path.join(save_video_path, clip[:-4])

    # if not os.path.exists(save_clip_path):
    #     os.makedirs(save_clip_path)

    # initialize the video writer
    video_writer = cv2.VideoWriter(save_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), handobj_fps, (image_size[0]*2, image_size[1]))

    # read the csv file
    df = pd.read_csv(clip_path)

    start_second = int(clip.split('.')[0])
    end_second = start_second + seconds
    start_frame = int(start_second * video_fps)
    end_frame = min(int(end_second * video_fps), len(video_reader))

    # sample seconds*handobj_fps frames
    sample_frames = np.linspace(start_frame, end_frame, num=len(df), endpoint=False, dtype=int)

    # # read the video frames
    # frames = video_reader.get_batch(sample_frames).asnumpy()

    for frame_id in range(len(df)):
        hand_image = crop_hands(video_reader, sample_frames, df, frame_id)

        # # save the frame as image
        # cv2.imwrite(os.path.join(save_clip_path, f'{frame_id:05d}.png'), hand_image)

        # write the frame to the video
        video_writer.write(hand_image)

    video_writer.release()
    print(f"Save {save_clip_path}")
    aa = 1

if __name__ == "__main__":
    video_root = "/mnt/SV_storage/VFM/EK100/EPIC-KITCHENS"
    handobj_root = "/mnt/SV_storage/VFM/hand_object_detector/Save_dir"
    save_path = "/mnt/SV_storage/VFM/hand_object_detector/hand_video"

    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    subjects = sorted(os.listdir(video_root))
    for subject in subjects[:37]:
        subject = 'P01'
        subject_path = os.path.join(video_root, subject, 'videos')
        save_subject_path = os.path.join(save_path, subject)
        if not os.path.exists(save_subject_path):
            os.makedirs(save_subject_path)

        videos = sorted(os.listdir(subject_path))
        for video in videos:
            video = 'P01_01.MP4'
            video_path = os.path.join(subject_path, video)
            handobj_path = os.path.join(handobj_root, subject, video)
            save_video_path = os.path.join(save_subject_path, video)
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path)
            
            clips = sorted(os.listdir(handobj_path))

            for clip_i in range(len(clips)):
                clip_i = 0
                process_clip(clips, video_path, handobj_path, save_video_path, clip_i)
                aa = 1

            
            # Parallel(n_jobs=6)(delayed(process_clip)(clips, video_path, handobj_path, save_video_path, clip_i) for clip_i in range(len(clips)))
                    

                