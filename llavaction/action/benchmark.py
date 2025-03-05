# benchmark gpt-4o on avion_mcq_top5_500
# benchmark gpt-4o on tim_mcq_top5_500
# benchmark gpt-4o on random_mcq_top5_500
from llavaction.action.chatgpt_utils import GPTInferenceAnnotator
import glob
import json
import os
import re

def process_raw_pred(raw_pred):
    matches = re.findall(r"[A-Z]\.\s(.+)", raw_pred)
    
    if 'None' in raw_pred:
        return raw_pred.replace('None. ', '')
    
    if matches:
        # Get the last match
        last_match = matches[-1]
        # Remove a trailing period and anything after it
        last_match = re.sub(r"\.\s*.*$", "", last_match)
        return last_match
    else:
        return raw_pred

# root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
# annotation_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
# avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'
# tim_prediction_file = '/data/epic_kitchen/TIM_PREDS/tim_pred_ids_val.json'

root = '/data/anonymous/EK100/'
annotation_file = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv'
avion_prediction_file = '/data/anonymous/AVION_PREDS/avion_pred_ids_val.json'
tim_prediction_file = '/data/anonymous/TIM_PREDS/tim_pred_ids_val.json'


n_frames = 8
topk = 5
action_representation = 'GT_random_narration'
perspective = 'first_person'
benchmark_testing = True


def benchmark_avion_mcq(n_samples, gpt_model, action_representation, benchmark_testing = True, n_frames = 8):

    inferencer = GPTInferenceAnnotator(gpt_model,
                                       root,
                                       annotation_file,
                                        gen_type = 'avion',
                                        prediction_file = avion_prediction_file,
                                        clip_length = n_frames,
                                        question_type = 'mc_',
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        topk = topk)
    inferencer.multi_process_run(n_samples = n_samples,
                                 offset = 0)
                                       
def benchmark_tim_mcq(n_samples, gpt_model, action_representation, benchmark_testing = True, n_frames = 8):
    
    inferencer = GPTInferenceAnnotator(gpt_model,
                                        root,
                                        annotation_file,
                                        gen_type = 'tim',
                                        prediction_file = tim_prediction_file,
                                        clip_length = n_frames,
                                        question_type = 'mc_',
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        topk = topk) 
    inferencer.multi_process_run(n_samples = n_samples, offset = 0)    

def benchmark_random_mcq(n_samples, gpt_model, action_representation, benchmark_testing = True, n_frames = 8):
    inferencer = GPTInferenceAnnotator(gpt_model,
                                       root,
                                       annotation_file,
                                        gen_type = 'random',
                                        prediction_file = avion_prediction_file,
                                        clip_length = n_frames,
                                        question_type = 'mc_',
                                        action_representation=action_representation,
                                        perspective = perspective,
                                        benchmark_testing = benchmark_testing,
                                        topk = topk) 
    
    inferencer.multi_process_run(n_samples = n_samples, offset = 0)
    
def calcuate_acc_from_jsons(json_folder):
    files = glob.glob(os.path.join(json_folder, '*.json'))
    for file in files:
        print (file)
        preds = json.load(open(file))
        correct = 0
        something = 0
        for k,v in preds.items():
            options = v['options']
            options = [process_raw_pred(e) for e in options]
            
            #assert v['gt_name'] in options, f"{v['gt_name']} not in {options}"
            if v['gt_name'] not in options:
                print ('what?', options)
                print ('what?', v)
                break
            
            if v['gt_name'] == v['chatgpt_answer']:
                correct+=1
            else:
                pass
                #print ('wrong prediction! pred: gt', v['chatgpt_answer'] + "," + v['gt_name'])
        print ('acc ', correct/len(preds))
        print ('gt not in options', something)

    
    
if __name__ == '__main__':
    # benchmark_avion_mcq(-1, 'gpt-4o-mini-2024-07-18', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    # benchmark_tim_mcq(-1, 'gpt-4o-mini-2024-07-18', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    # benchmark_random_mcq(-1, 'gpt-4o-mini-2024-07-18', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    # benchmark_avion_mcq(-1, 'gpt-4o-2024-08-06', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    # benchmark_tim_mcq(-1, 'gpt-4o-2024-08-06', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    # benchmark_random_mcq(-1, 'gpt-4o-2024-08-06', 'GT_random_narration', benchmark_testing = True, n_frames = 8)
    benchmark_tim_mcq(1, 'gpt-4o-mini-2024-07-18', 'official_key', benchmark_testing = False, n_frames = 16)
    #benchmark_tim_mcq(-1, 'gpt-4o-mini-2024-07-18', 'GT_random_narration', benchmark_testing = False, n_frames = 16)
    #calcuate_acc_from_jsons('gpt_EK100_results')