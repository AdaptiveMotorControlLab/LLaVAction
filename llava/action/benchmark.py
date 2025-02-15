# benchmark gpt-4o on avion_mcq_top5_500
# benchmark gpt-4o on tim_mcq_top5_500
# benchmark gpt-4o on random_mcq_top5_500
from llava.action.chatgpt_utils import GPTInferenceAnnotator
import glob
import json
import os
# root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
# annotation_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
# avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'
# tim_prediction_file = '/data/epic_kitchen/TIM_PREDS/tim_pred_ids_val.json'

root = '/data/shaokai/EK100/'
annotation_file = '/data/shaokai/epic-kitchens-100-annotations/EPIC_100_validation.csv'
avion_prediction_file = '/data/shaokai/AVION_PREDS/avion_pred_ids_val.json'
tim_prediction_file = '/data/shaokai/TIM_PREDS/tim_pred_ids_val.json'


n_frames = 8
topk = 5
action_representation = 'GT_random_narration'
perspective = 'first_person'
benchmark_testing = True


def benchmark_avion_mcq(n_samples, gpt_model):

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
                                       
def benchmark_tim_mcq(n_samples, gpt_model):
    
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

def benchmark_random_mcq(n_samples, gpt_model):
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
        for k,v in preds.items():
            if v['gt_name'] == v['chatgpt_answer']:
                correct+=1
        print ('acc ', correct/len(preds))

    
    
if __name__ == '__main__':
    # benchmark_avion_mcq(-1, 'gpt-4o-mini-2024-07-18')
    # benchmark_tim_mcq(-1, 'gpt-4o-mini-2024-07-18')
    # benchmark_random_mcq(-1, 'gpt-4o-mini-2024-07-18')
    # benchmark_avion_mcq(-1, 'gpt-4o-2024-08-06')
    # benchmark_tim_mcq(-1, 'gpt-4o-2024-08-06')
    # benchmark_random_mcq(-1, 'gpt-4o-2024-08-06')    
    calcuate_acc_from_jsons('gpt_full_benchmark_results')