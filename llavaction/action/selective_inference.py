"""
Instead of running the whole validation set, 
"""
from llavaction.action.ek_eval import prepare_llava
from llavaction.action.generate_interval_pred import  get_lookup_dict
from llavaction.action.llava_inference import llava_inference
from llavaction.action.utils import avion_video_loader

from llavaction.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

val_metadata = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv'  
data_root = '/data/anonymous/EK100_512/EK100'

n_frames = 32
action_representation = 'GT_random_narration'
perspective = 'first_person'
include_time_instruction = False
image_token = DEFAULT_IMAGE_TOKEN



def get_frames_by_uid(uid, root):
    vid_path = '_'.join(uid.split('_')[:2]).replace('-', '/')
    print ('debug', uid)
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
    return frames, time_meta   
#

 
                        
                        
                        

# for prior actions
def get_meta_data():
    pass


def inference_task_by_uid(data_root, question, checkpoint_folder, uid, task):
    
    tokenizer, model, image_processor, max_length = prepare_llava(checkpoint_folder)
    
    frames, time_meta = get_frames_by_uid(uid, data_root)
    
    meta_data = None
    learn_neighbor_actions = ""
    if 'temporal_cot' in task:
        lookup_table = get_lookup_dict(val_metadata, 
                        action_representation,
                        test_type = task, 
                        pseudo_folder = '')
        meta_data = lookup_table.get(uid, None)
        learn_neighbor_actions = "prior"
    
    video_duration = time_meta['duration']
            
        
    pred = llava_inference(
                        [frames], 
                        tokenizer, 
                        model, 
                        image_processor,  
                        question,  
                        test_type = task,
                        clip_length = n_frames, 
                        num_frames= n_frames, 
                        temperature = 0,
                        time_meta = time_meta,
                        learn_neighbor_actions = learn_neighbor_actions,
                        meta_data = meta_data,
                        perspective = perspective,
                        include_time_instruction = include_time_instruction
                        )
    return pred
    
class SelectiveInferencer:
    def __init__(self, data_root, checkpoint_folder, include_time_instruction = False, n_frames = 32, use_flash_attention = True):
        self.data_root = data_root
        self.checkpoint_folder = checkpoint_folder
        self.tokenizer, self.model, self.image_processor, self.max_length = prepare_llava(checkpoint_folder, use_flash_attention = use_flash_attention)
        self.include_time_instruction = include_time_instruction
        self.n_frames = n_frames
    def inference(self, question, uid, task):
        frames, time_meta = get_frames_by_uid(uid, self.data_root)
        
        meta_data = None
        learn_neighbor_actions = ""
        if 'temporal_cot' in task:
            lookup_table = get_lookup_dict(val_metadata, 
                            action_representation,
                            test_type = task, 
                            pseudo_folder = '')
            meta_data = lookup_table.get(uid, None)
            learn_neighbor_actions = "prior"
        
                        
        pred = llava_inference(
                            [frames], 
                            self.tokenizer, 
                            self.model, 
                            self.image_processor,  
                            question,  
                            test_type = task,
                            clip_length = self.n_frames, 
                            num_frames= self.n_frames, 
                            temperature = 0,
                            time_meta = time_meta,
                            learn_neighbor_actions = learn_neighbor_actions,
                            meta_data = meta_data,
                            perspective = perspective,
                            include_time_instruction = self.include_time_instruction
                            )
        return pred        
        
    
if __name__ == '__main__':
    pretrained_model_folder = 'experiments/dev_LLaVA-Video-7B-Qwen2'
    uid = 'P28-P28_15_50.66_51.69'
    task = 'open-ended'
    question = "What is the object that is to the left of the knife?"
    
    inference_task_by_uid(data_root, 
                          question,
                          pretrained_model_folder,
                          uid,
                          task)