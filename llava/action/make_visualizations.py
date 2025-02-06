"""
We need to keep track of the following:

The uid of each segment

The GPT inference of corresponding segment
The LLaVA zero-shot inference of corresponding segment
The Finetuned LLaVA's inference of corresponding segment

Note that in each inference, we should be able to pick the corresponding prompt and checkpoint folder
"""

from llava.action.chatgpt_utils import GPTInferenceAnnotator

root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
annotation_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'
tim_prediction_file = '/data/epic_kitchen/TIM_PREDS/tim_pred_ids_val.json'
n_frames = 4
topk = 5
action_representation = 'GT_random_narration'
gpt_model = 'gpt-4o-mini-2024-07-18'
#gpt_model = 'gpt-4o-2024-08-06'
perspective = 'first_person'
benchmark_testing = True



def visualize_with_random(n_samples, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
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
    
    inferencer.multi_process_run(n_samples, disable_api_calling=False)

def visualize_with_gpt_with_tim(n_samples, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
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
    
    inferencer.multi_process_run(n_samples, disable_api_calling=False)    


def visualize_with_gpt_with_avion(n_samples, question_type = 'mc_'):
    """
    Here we should test gpt-4o, gpt-4o-mini with different prompts
    """
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
    
    inferencer.multi_process_run(n_samples, disable_api_calling=False) 

if __name__ == '__main__':
    #visualize_with_random(1, question_type = "mc_")
    #visualize_with_gpt_with_tim(1, question_type = "mc_")
    visualize_with_gpt_with_avion(1, question_type = "mc_")
