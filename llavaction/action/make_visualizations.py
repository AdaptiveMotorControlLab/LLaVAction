"""
We need to keep track of the following:

The uid of each segment

The GPT inference of corresponding segment
The LLaVA zero-shot inference of corresponding segment
The Finetuned LLaVA's inference of corresponding segment

Note that in each inference, we should be able to pick the corresponding prompt and checkpoint folder
"""
from llavaction.action.utils import generate_label_map  

from pathlib import Path
from llavaction.action.utils import AvionMultiChoiceGenerator as ActionMultiChoiceGenerator
from llavaction.action.llava_inference import llava_inference
import json
import cv2

root = '/data/anonymous/EK100_512/EK100'
annotation_file = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv'
avion_prediction_file = '/data/anonymous/AVION_PREDS/avion_pred_ids_val.json'
tim_prediction_file = '/data/anonymous/TIM_PREDS/tim_pred_ids_val.json'
val_metadata = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv' 

n_frames = 32
topk = 5
action_representation = 'GT_random_narration'
gpt_model = 'gpt-4o-mini-2024-07-18'
#gpt_model = 'gpt-4o-2024-08-06'
perspective = 'first_person'
benchmark_testing = True


def visualize_with_random(n_samples, offset = 0, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
    from llavaction.action.chatgpt_utils import GPTInferenceAnnotator
    inferencer = GPTInferenceAnnotator(gpt_model,
                                       root,
                                       annotation_file,
                                        gen_type = 'random',
                                        prediction_file = tim_prediction_file,
                                        clip_length = n_frames,
                                        question_type = question_type,
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        do_visualization = True,
                                        topk = topk) 
    
    inferencer.multi_process_run(n_samples = n_samples, offset = offset, disable_api_calling=False)

def visualize_with_gpt_with_tim(n_samples, offset = 0, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
    from llavaction.action.chatgpt_utils import GPTInferenceAnnotator
    inferencer = GPTInferenceAnnotator(gpt_model,
                                       root,
                                       annotation_file,
                                        gen_type = 'tim',
                                        prediction_file = tim_prediction_file,
                                        clip_length = n_frames,
                                        question_type = question_type,
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        do_visualization = True,
                                        topk = topk) 
    
    inferencer.multi_process_run(n_samples = n_samples, offset = offset, disable_api_calling=False)    


def visualize_with_gpt_with_avion(n_samples, offset = 0, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
    from llavaction.action.chatgpt_utils import GPTInferenceAnnotator
    inferencer = GPTInferenceAnnotator(gpt_model,
                                       root,
                                       annotation_file,
                                        gen_type = 'avion',
                                        prediction_file = avion_prediction_file,
                                        clip_length = n_frames,
                                        question_type = question_type,
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        do_visualization = True,
                                        topk = topk) 
    
    inferencer.multi_process_run(n_samples = n_samples, offset = offset, disable_api_calling=False) 
    
    
def search_option_data_by_uid(uid, anno_file, gen_type = 'tim'):
    import csv
    from llavaction.action.dataset import datetime2sec
    csv_reader = csv.reader(open(anno_file, 'r'))
    _ = next(csv_reader) # skip the header
    query_vid_path = '_'.join(uid.split('_')[:2]).replace('-', '/')
    query_start_timestamp, query_end_timestamp = uid.split('_')[2:]
    anno_root = Path(anno_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            action_representation)    
    with open(tim_prediction_file, 'r') as f:
        action_model_predictions = json.load(f)
    mc_generator = ActionMultiChoiceGenerator(anno_root)
    
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]       
        
        if query_vid_path!=vid_path and start_second!=query_start_timestamp and end_second!=query_end_timestamp:
            continue
        
        if gen_type == 'avion' or gen_type == 'tim':
            action_preds = action_model_predictions[str(idx)]['predictions']
            mc_data =mc_generator.generate_multi_choice(gt_vn,
                                                        action_preds,
                                                        narration,
                                                        topk,
                                                        action_representation,
                                                        -1, # n_narrations
                                                        labels,
                                                        mapping_vn2narration,
                                                        verb_maps,
                                                        noun_maps,
                                                        benchmark_testing = benchmark_testing,
                                                        is_train = False)            
            
            options = mc_data['options'][0]
            return {
                'options': options,
                'narration': narration,
                'start_second': start_second,
                'end_second': end_second,
                'gt_answer': gt_vn
            }
    
def save_visualization(vis_folder, frames, uid): 
    out_dir = Path(vis_folder)
    out_dir.mkdir(parents=True, exist_ok=True)        
    sub_folder = out_dir / uid
    fps = 30
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = str(sub_folder / f"{uid}.mp4")
    video_out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    sub_folder.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):            
        cv2.imwrite(str(sub_folder / f"{uid}_{idx}.jpg"), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))    
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_out.write(bgr_frame)
    video_out.release()

def visualize_with_uid(data_root, uid, out_folder):
    from llavaction.action.utils import avion_video_loader
                   
    vid_path = '_'.join(uid.split('_')[:2]).replace('-', '/')
    start_timestamp, end_timestamp = uid.split('_')[2:]
    start_timestamp = float(start_timestamp)
    end_timestamp = float(end_timestamp)
    print (vid_path, start_timestamp, end_timestamp)
    # split uid to video path and start, end second
    frames, time_meta = avion_video_loader(data_root,
                                           vid_path,
                                           'MP4',
                                            start_timestamp,
                                            end_timestamp,
                                            chunk_len = 15,
                                            clip_length = n_frames,
                                            threads = 1,
                                            fast_rrc=False,
                                            fast_rcc = False,
                                            jitter = False)
    

    save_visualization(out_folder, frames, uid)  
    return frames     
    
def visualize_with_llava(pretrained_path, uid, question_type, gen_type):
    """    
    """
    from llavaction.action.ek_eval import prepare_llava
    from llavaction.action.dataset import VideoMultiChoiceDataset
  
    import torch
    
    from llavaction.action.utils import avion_video_loader
    val_metadata = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv'
        
    gpu_val_transform_ls = []

    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)
        
    vid_path = '_'.join(uid.split('_')[:2]).replace('-', '/')
    start_timestamp, end_timestamp = uid.split('_')[2:]
    start_timestamp = float(start_timestamp)
    end_timestamp = float(end_timestamp)
    print (vid_path, start_timestamp, end_timestamp)
    # split uid to video path and start, end second
    frames, time_meta = avion_video_loader(root,
                                           vid_path,
                                           'MP4',
                                            start_timestamp,
                                            end_timestamp,
                                            chunk_len = 15,
                                            clip_length = n_frames,
                                            threads = 1,
                                            fast_rrc=False,
                                            fast_rcc = False,
                                            jitter = False)
    
    vis_folder = f"{gpt_model}_{gen_type}_{question_type}_{perspective}"                       
    save_visualization(vis_folder, frames, uid)                       
                                            
    options = search_option_data_by_uid(uid, val_metadata, gen_type = gen_type)
    
    print (options)                                 
    mc_data = options                           
    tokenizer, model, image_processor, _ = prepare_llava(pretrained_path)      
    pred = llava_inference(
                            [frames], 
                            tokenizer, 
                            model, 
                            image_processor,  
                            mc_data,  
                            test_type = question_type,
                            clip_length = n_frames, 
                            num_frames=n_frames,
                            temperature = 0,
                            time_meta = time_meta,
                            learn_neighbor_actions = "",
                            meta_data = None,
                            perspective = perspective
                            )
    
    print (pred)
if __name__ == '__main__':
    
    visualize_with_uid(root, 'P23-P23_05_217.41_218.39', 'figure1_vis')  