import base64
import io
import json
import os
import numpy as np
import openai
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from action.utils import avion_video_loader, create_multi_choice_from_avion_predictions
import torch
import cv2
from pathlib import Path
from action.prediction_analysis import PredictionAnalysis

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-4o-2024-08-06"

prices = {
    "gpt-4o-2024-08-06": {"input": 2.5 / 10**6, "output": 10 / 10**6},
}

class GT_Agnostic_Response(BaseModel):
    """
    The GT was not known. The response is to generate a new answer
    """
    explanation: str
    answer: str

class GT_Augmentation_Response(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    explanation: str


class ChatGPT:
    """
    Importantly, this class should handle the error in case the inference fails in the middle    
    """

    def __init__(self, clip_length = 4):
        self.clip_length = clip_length

    def checkpoint(self):
        """
        In case we fail in the middle, we can still restore the progress
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def multi_process_run(self):
        """
        This function should split the data and run the inference in parallel
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def run(self, indices):
        """
        This function should run the inference in a subset of the whole data.
        This is to support multi-processing job
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def checkpoint(self, results, out_path):
        print ('saving checkpoint to ', out_path)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent = 4)

    def resume_from_checkpoint(self, checkpoint_path):
        pass

    def calculate_cost(self, response):
        input_consumed = response.usage.prompt_tokens
        output_consumed = response.usage.completion_tokens
        input_cost = input_consumed * prices[GPT_MODEL]["input"]
        output_cost = output_consumed * prices[GPT_MODEL]["output"]
        total_cost = input_cost + output_cost
        print (f'cost of the inference {total_cost:.4f}')
        return total_cost

    def split_indices(self, indices, num_chunks):
        """
        Split the indices into num_chunks
        """
        chunk_size = len(indices) // num_chunks
        remainder = len(indices) % num_chunks

        chunks = []
        start = 0
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(indices[start:end])
            start = end

        return chunks


    def prepare_multiple_images(self, images):
        """

        """               
        encoded_image_list = []

        for image in images:
         
            if isinstance(image, torch.Tensor):
                image = image.cpu().detach().numpy()

            
            # images from matplotlib etc.
            if isinstance(image, io.BytesIO):
                image_bytes = image
                base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            # images from opencv
            elif isinstance(image, np.ndarray):
                result, buffer = cv2.imencode(".jpeg", image)
                image_bytes = io.BytesIO(buffer)
                base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

            encoded_image_list.append(base64_image)

        multi_image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
            for encoded_image in encoded_image_list
        ]

        return multi_image_content 

    def extract_frames(self,  vid_path, start_second, end_second):
        frames, time_meta = avion_video_loader(self.root,
                        vid_path,
                        'MP4',
                        start_second,
                        end_second,
                        chunk_len = 15,
                        clip_length = self.clip_length,
                        threads = 1,
                        fast_rrc=False,
                        fast_rcc = False,
                        jitter = False)
        return frames, time_meta               


class GPTInferenceAnnotator(ChatGPT):
    """
    Given the images, this class will annotate the video frames
    """

    def __init__(self, root, prediction_save_folder, clip_length = 4, debug = False):
        super().__init__(clip_length = clip_length)
        self.root = root
        self.prediction_save_folder = prediction_save_folder 
        self.prediction_analysis = PredictionAnalysis(self.prediction_save_folder)
        self.prediction_analysis.load()
        self.data = self.prediction_analysis.data
        self.debug = debug

    def multi_process_run(self):
        prediction_analysis = PredictionAnalysis(self.prediction_save_folder)
        # to initialize it
        prediction_analysis.load()
        data = prediction_analysis.data

        indices = list(range(len(data)))

        num_chunks = os.cpu_count() if not self.debug else 2

        indices_groups = self.split_indices(indices, num_chunks)

        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            # Pass additional arguments to the function
            futures = [executor.submit(self.run, group) for group in indices_groups]
            
            # Wait for all futures to complete
            combined_results = {}
            for future in futures:
                result_dict = future.result()
                combined_results.update(result_dict)

        self.checkpoint(combined_results, "gpt_inference_results.json")
                        

    def parse_item(self, item):

        gt_name = item['gt_name']
        avion_predictions = item['avion_preds']['predictions']
        # _avion_predictions = [e.replace(':', ' ', 1) for e in avion_predictions]
        # if gt_name not in _avion_predictions:
        #     print ('gt_name not in avion_predictions')
        # else:
        #     print ('gt_name in avion_predictions')
        
        vid_path = item['vid_path'][0]
        start_second = item['start_second']
        end_second = item['end_second']
    
        frames, time_meta = self.extract_frames(vid_path, start_second, end_second)

        option_text = create_multi_choice_from_avion_predictions(avion_predictions, len(avion_predictions))['options'][0]
        parsed_item = {
            'options': option_text,
            'gt_answer': gt_name,
            'start_second': start_second,
            'end_second': end_second,
            'vid_path': vid_path
        }

        return parsed_item                   
        

    def run(self, indices):
        data_batch = {i : self.data[i] for i in range(len(self.data)) if i in indices}
        ret = {}

        for k,v in data_batch.items():
            parsed_item = self.parse_item(v)
            start_timestamp = parsed_item['start_second']
            end_timestamp = parsed_item['end_second']
            vid_path = parsed_item['vid_path']

            frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)
            try:
                parsed_answer = self.predict_images(frames, parsed_item)
            except Exception as e:
                print ("An exception occurred: ", e)
            predicted_answer = parsed_answer.answer
            explanation = parsed_answer.explanation
            gt_name = parsed_item['gt_answer']
            ret[k] = {
                'gt_name': gt_name,
                'chatgpt_answer': predicted_answer
            }
            if self.debug:
                break
        print (ret)
        return ret 

    def explore_wrong_examples(self):
        wrong_examples = self.prediction_analysis.get_wrong_examples()
        ret = {}
        count = 0
        for k,v in wrong_examples.items():                     
            parsed_item = self.parse_item(v)

            start_timestamp = parsed_item['start_second']
            end_timestamp = parsed_item['end_second']
            vid_path = parsed_item['vid_path']
            gt_name = parsed_item['gt_answer']
            frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)

            parsed_answer = self.predict_images(frames, parsed_item)
            
            predicted_answer = parsed_answer.answer
            explanation = parsed_answer.explanation

            print ('gt_name', gt_name)
            print ('avion_predictions', v['avion_preds']['predictions'])
            print ('llava_pred', v['llava_pred'])
            print ('chatgpt answer', predicted_answer)
            print ('explanation', explanation)

            ret[k] = {
                'gt_name': gt_name,
                'avion_predictions': v['avion_preds']['predictions'],
                'llava_pred': v['llava_pred'],
                'chatgpt_answer': predicted_answer,
                'explanation': explanation
            }                  
            if self.debug:
                break


        return ret

    def predict_images(self, images, parsed_item):
        """
        Predict the action from the images
        """
        
        option_text = parsed_item['options']
        start_second = parsed_item['start_second']
        end_second = parsed_item['end_second']
        temperature = 0
        system_prompt_prefix = f"""
        You are seeing video frames from an egocentric view of a person. Pretend that you are the person.  Your task is to describe what action you are performing.
        To assist you for how to describe the action, the video's start time is {start_second} and the end time is {end_second} and the duration is {end_second - start_second} seconds.
        You were given multiple choice options {option_text}. Pick the correct one and put that into the answer. Note in the answer do not include the option letter, just the name of the action.
        Also explain why the correct answer is correct and why the other options are incorrect.
        """

        system_prompt_suffix = """"""

        system_prompt = system_prompt_prefix + system_prompt_suffix

        system_message =  [{"role": "system", "content": system_prompt}]

        multi_image_content = self.prepare_multiple_images(images)
        multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
        user_message = [{"role": "user", "content": multi_modal_content}]               

        response = client.beta.chat.completions.parse(
            model=GPT_MODEL,
            messages=system_message + user_message, 
            response_format = GT_Agnostic_Response,
            temperature = temperature
        )
        total_cost = self.calculate_cost(response)

        return response.choices[0].message.parsed


class GPTAugmentationAnnotator(ChatGPT):
    """
    Given the train annotation from the EK100 dataset, this class will annotate the video frames
    that augments the gt annotations.
    """

    def __init__(self, ann_file, root, clip_length = 4, debug = False):
        super().__init__(clip_length = clip_length) 
        self.ann_file = ann_file
        self.root = root
        self.clip_length = clip_length
        data = []
        with open(ann_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.data = data
        self.debug = debug

    def parse_conversation_from_train_convs(self, item):
        """
        The item has the structure of convs defined in the train anno.
        """
        conversations = item['conversations']
        human_dict = conversations[0]

        # the offset is to remove the quote ' 
        option_start = human_dict['value'].index('[') + 2
        option_end = human_dict['value'].index(']') - 1

        option_text =  human_dict['value'][option_start:option_end]        
        gpt_dict = conversations[1]
        gt_answer = gpt_dict['value']
        gt_answer = gt_answer[gt_answer.index('.'):].strip()

        assert human_dict['from'] == 'human' and gpt_dict['from'] =='gpt'

        ret = {'options': option_text,
               'gt_answer': gt_answer,
               'start_second': item['start_timestamp'],
               'end_second':  item['end_timestamp']}
        
        return ret

    def multi_process_run(self):
        indices = list(range(len(self.data)))
        num_cores = os.cpu_count() if not self.debug else 2
        indices_groups = self.split_indices(indices, num_cores)

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Pass additional arguments to the function
            futures = [executor.submit(self.run, group) for group in indices_groups]
            
            # Wait for all futures to complete
            combined_results = {}
            for future in futures:
                result_dict = future.result()
                combined_results.update(result_dict)

        self.checkpoint(combined_results, "gpt_annotated.json")
        print ('finished the annotation')
        return combined_results

    def run(self, indices):
        data_batch = [self.data[i] for i in range(len(self.data)) if i in indices]

        ret = {}
        for index in indices:
            item = self.data[index]
            start_timestamp = item['start_timestamp']
            end_timestamp = item['end_timestamp']
            vid_path = '{}/{}'.format(item['video'].split('-')[0], item['video'].split('-')[1])
            frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)
            parsed_item = self.parse_conversation_from_train_convs(item)
            try:
                gpt_answer = self.annotate(frames, parsed_item).explanation
            except Exception as e:
                print ("An exception occurred: ", e)
                continue
            item['conversations'][1]['value'] = gpt_answer
            ret[index] = item
            if self.debug:
                break

        return ret       

    def annotate(self, images, data_item):
        """
        Assuming that data_item already has the multi-choice options and the gt_answer
        """
        gt_answer = data_item['gt_answer']
        option_text = data_item['options']
        start_second = data_item['start_second']
        end_second = data_item['end_second']        
        temperature = 0
        system_prompt_prefix = f"""
You are seeing video frames from an egocentric view of a person. 
Please talk as if you are the person in the video and describe what action you are performing.
To assist you for how to describe the action, the video's start time is {start_second} and the end time is {end_second} and the duration is {end_second - start_second} seconds.
To further assist you for how to describe the action, note that in a multi-choice video question answering, you were given following options {option_text} and the correct answer is {gt_answer}.
In addition to describe what you see, why wrong answers were wrong and why right answer was right.
When you explain why wrong answers were wrong and why right answer was right, you should use the following flow of reasoning:

The flow of reasoning:
1. What objects need to be visible to support the answer?
2. What sequence of actions before and after the current action need to be seen to support the answer?
3. Whether the duration in time supports that answer?

Based on the answers above, why right answer is right and why wrong answers were wrong.


"""
        system_prompt_suffix = """"""

        system_prompt = system_prompt_prefix + system_prompt_suffix

        system_message =  [{"role": "system", "content": system_prompt}]

        multi_image_content = self.prepare_multiple_images(images)
        multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
        user_message = [{"role": "user", "content": multi_modal_content}]               

        response = client.beta.chat.completions.parse(
            model=GPT_MODEL,
            messages=system_message + user_message, 
            response_format = GT_Augmentation_Response,
            temperature = temperature
        )
        total_cost = self.calculate_cost(response)

        return response.choices[0].message.parsed
    

def multi_process_annotate(train_file_path, root):
    annotator = GPTAugmentationAnnotator(train_file_path, 
    root, 
    clip_length = 4,
    debug = True)
    results = annotator.multi_process_run()

def explore_wrong_examples(root, prediction_save_folder):
    annotator = GPTInferenceAnnotator(root, 
                                    prediction_save_folder, 
                                    clip_length = 4, 
                                    debug = True)
    annotator.explore_wrong_examples()

def multi_process_inference(root, prediction_save_folder):

    annotator = GPTInferenceAnnotator(root, 
    prediction_save_folder, 
    clip_length = 4,
    debug = True)
    annotator.multi_process_run()

if __name__ == '__main__':
    #train_file_path = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train/avion_mc_top10/train_convs_narration.jsonl'
    #root = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100'
    train_file_path = '/data/EK100_inst_train/avion_mc_top10/train_convs_narration.jsonl'
    root = '/data/EK100/EK100_320p_15sec_30fps_libx264'    
    pred_folder = '/data/epic_kitchen/llavavideo_avion_mc_top10_5epoch_preds'

    multi_process_annotate(train_file_path, root)
    #explore_wrong_examples(root, pred_folder)

    #multi_process_inference(root, pred_folder)
