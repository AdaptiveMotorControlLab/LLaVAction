# benchmark gpt-4o on avion_mcq_top5_500
# benchmark gpt-4o on tim_mcq_top5_500
# benchmark gpt-4o on random_mcq_top5_500
from llava.action.chatgpt_utils import GPTInferenceAnnotator

root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
annotation_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'
tim_prediction_file = '/data/epic_kitchen/TIM_PREDS/tim_pred_ids_val.json'
n_frames = 4
topk = 5
action_representation = 'GT_random_narration'
#gpt_model = 'gpt-4o-mini-2024-07-18'
gpt_model = 'gpt-4o-2024-08-06'
perspective = 'first_person'
benchmark_testing = True


def benchmark_avion_mcq(n_samples):

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
    inferencer.multi_process_run(n_samples)
                                       
def benchmark_tim_mcq(n_samples):
    
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
    inferencer.multi_process_run(n_samples)    

def benchmark_random_mcq(n_samples):
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
    
    inferencer.multi_process_run(n_samples)
    
    
if __name__ == '__main__':
    #benchmark_avion_mcq(100)
    benchmark_tim_mcq(100)
    #benchmark_random_mcq(100)    