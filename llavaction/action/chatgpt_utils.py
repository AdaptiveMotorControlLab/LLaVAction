import os
import sys
sys.path[0] = os.path.dirname(os.path.dirname(sys.path[0]))
import openai
from pydantic import BaseModel
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from llavaction.action.utils import AvionMultiChoiceGenerator as ActionMultiChoiceGenerator
from llavaction.action.utils import RandomMultiChoiceGenerator
from llavaction.action.utils import avion_video_loader, avion_video_render_loader, generate_label_map
from llavaction.action.dataset import datetime2sec
from llavaction.action.ek_eval import process_raw_pred
import csv
import copy 
import torch
import io
import numpy as np 
import base64
from pathlib import Path
import traceback
import cv2


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-4o"

prices = {
    "gpt-4o": {"input": 2.5 / 10**6, "output": 10 / 10**6},
    "gpt-4o-2024-08-06": {"input": 2.5 / 10**6, "output": 10 / 10**6},
    "o1": {"input": 15 / 10**6, "output": 60 / 10**6},
    "o1-mini": {"input": 3 / 10**6, "output": 12 / 10**6},
    "gpt-4o-mini": {"input": 0.15 / 10**6, "output": 0.6 / 10**6},
    "gpt-4o-mini-2024-07-18": {"input": 0.15 / 10**6, "output": 0.6 / 10**6},
    
}

class ExpandReasonMCPrompt:
    """
    Given the reasoning + mc description, create multiple questions
    The questions include 
    1) Why wrong answers are wrong
    2) Other questions that can be asked given the reasoning
    """
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):

        reason_mc_string = gt_answer 

        prompt = f"""Your job is to create 3 question-answer pairs based on the text below. The text contains a first-person narrative of video frames from an egocentric perspective of a person interacting with objects in a kitchen.
{reason_mc_string}
You can ask questions such as:
What object am I interacting with?
What objects are visible in the video?
What is the sequence of the atomic actions I am performing?

Make sure your questions can be answered based on the information provided in the text. Do not ask questions that require additional context or information beyond what is given.

        """
        return prompt


class GPTHandObjectPrompt:
    @classmethod
    def generate_prompt(cls, left_hand_state, right_hand_state, gt_narration):
        prompt = f""" 
You are a helpful AI assistant, and you will assist in creating question-answer pairs.
I will provide you with the state of the left hand and the right hand, as well as the ground-truth narration.
For the hand states:
- -1 denotes the hand is not visible
- 0 denotes the hand is visible but not interacting with objects
- 1 denotes the hand is interacting with another hand
- 3 denotes the hand is interacting with a portable object
- 4 denotes the hand is interacting with a stationary object

The state for the left hand is {left_hand_state}, and the state for the right hand is {right_hand_state}.
Using this information, create 3 question-answer pairs. Pretend you are seeing an image from a first-person perspective and can see your hands and the objects you are interacting with.
Do not ask questions about the action, as you are viewing an image and not a video.
Do not describe what the object is, only mention whether it's portable or stationary.
Ask and answer the questions in the first-person perspective.
        """
        return prompt

class GPTReasoningWithGTPrompt:
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):
        prompt = f"""
You are viewing video frames from an egocentric perspective and you are the person. Describe the video frames in detail and reason about the actions you are performing. You will be provided with the human-annotated ground-truth for the action, but you should independently come to your own conclusion.
If you disagree with the human annotation, indicate "true" in the "disagree_with_human_annotation" field of your response, and provide your reasoning without mentioning the ground-truth answer. This will keep your reasoning clean. If you agree with the human annotation, indicate "false" in the "disagree_with_human_annotation" field and provide your reasoning without referencing the ground-truth to maintain a clean description.

The true ground-truth action is {gt_answer}.
Your reasoning steps should include supporting evidence for the action, such as the duration of the video, the sequence of actions the person performs, the objects they interact with, and the overall context of the video.
As a general guideline, for videos longer than 3 seconds, provide detailed reasoning steps, and for videos shorter than 3 seconds, generate less detailed reasoning.
The video duration is {end_second - start_second:.3f} seconds.
Make sure you use the first-person perspective in your reasoning.
"""
        print (prompt)
        return prompt


class InferenceAnswer:
    json_errors = 0
    def __init__(self, answer, gpt_model):
        
        if 'o1' not in gpt_model:
            self.answer = answer.answer
        else:
            content = answer.content
            temp = content.replace('```json', '').replace('```', '').strip()            
            try:
                answer = json.loads(temp)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response: {response_content}")
                self.answer = 'N/A'
                json_errors += 1
            self.answer = answer['answer']
            


class GPTStrongReasoningWithGTPrompt:
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):
        prompt = f"""
You are viewing video frames from an egocentric perspective of and you are the person. Sometimes you interact with objects. Describe the video frames in detail and reason about the actions the person is performing. You will be provided with the human-annotated ground-truth for the action, but you should independently come to your own conclusion.

Your reasoning steps should include supporting evidence for the action, such as the duration of the video, the sequence of actions the person performs, the objects they interact with, and the overall context of the video.
As a general guideline, for videos longer than 3 seconds, provide detailed reasoning steps, and for videos shorter than 3 seconds, generate less detailed reasoning.

The video duration is {end_second - start_second:.3f} seconds.

To summarize, in the field 'answer_with_reasoning':

    Explain why wrong answers are incorrect and provide additional reasoning for the correct answer. Your final response should look like [your reasoning thoughts] .. the correct answer is [your answer]  \n
    Make sure the multiple-choice answer follows the format of uppercase letter.  answer such as "A. move plate" 

And in the field 'disagree_with_human_annotation':

    If you disagree with the human annotation, indicate "true" in the "disagree_with_human_annotation" field of your response, and provide your reasoning without mentioning the ground-truth answer. This will keep your reasoning clean. If you agree with the human annotation, indicate "false" in the "disagree_with_human_annotation" field and provide your reasoning without referencing the ground-truth to maintain a clean description.

The candidate actions you are selecting from are {option_text}.  The true ground-truth action is {gt_answer}.

Make sure you use the first-person perspective in your reasoning.

"""
        print (prompt)
        return prompt        

class GT_Augmentation_Response(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    answer_with_reasoning: str
    disagree_with_human_annotation: bool


class MultiChoice_Response(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    answer: str


class GT_Agnostic_Response(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    answer: str
    caption: str


class GPTHandObjectResponse(BaseModel):
    """
    The response for the GPTHandObjectPrompt
    """
    first_question: str
    first_answer: str
    second_question: str
    second_answer: str
    third_question: str
    third_answer: str

class ExpandReasonMCResponse(BaseModel):
    """
    The response for the ExpandReasonMCPrompt
    """
    first_question: str
    first_answer: str
    second_question: str
    second_answer: str
    third_question: str
    third_answer: str

PROMPT_FACTORY = {'gpt-gt-reason': GPTReasoningWithGTPrompt,
                    'gpt-gt-strong-reason': GPTStrongReasoningWithGTPrompt,
                   'gpt-gt-instruct-reason': ExpandReasonMCPrompt,
                   'gpt-hand-object': GPTHandObjectPrompt}

REQUIRES_VIS = set(['gpt-gt-reason', 'gpt-gt-strong-reason'])

RESPONSE_FACTORY = {'gpt-gt-reason': GT_Augmentation_Response,
                    'gpt-gt-strong-reason': GT_Augmentation_Response,
                    'gpt-gt-instruct-reason': ExpandReasonMCResponse,
                    'gpt-hand-object': GPTHandObjectResponse}

class ChatGPT:
    """
    Importantly, this class should handle the error in case the inference fails in the middle    
    """

    def __init__(self, gpt_model, clip_length = 4):
        self.gpt_model = gpt_model
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
        input_cost = input_consumed * prices[self.gpt_model]["input"]
        output_cost = output_consumed * prices[self.gpt_model]["output"]
        total_cost = input_cost + output_cost
        #print (f'cost of the inference {total_cost:.4f}')
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
        import cv2               
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

    def extract_frames(self, vid_path, start_second, end_second):
        if hasattr(self, 'handobj_root') and self.handobj_root is not None:

            frames, time_meta = avion_video_render_loader(self.root, self.handobj_root,
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
                
        else:
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
    This class should also optionally take conversion map as we find that 
    there are multiple ways to map verb_id and noun_id.
    """

    def __init__(self, 
                 gpt_model,
                 root,                 
                 annotation_file,
                 gen_type = 'tim',
                 prediction_file = None,
                 handobj_root = None,
                 clip_length = 4, 
                 action_representation = 'GT_random_narration',
                 question_type = 'cot_mc',
                 debug = False,
                 topk = 10,
                 perspective = 'first_person',
                 benchmark_testing = False,
                 do_visualization = False
                 ):
        """
        Parameters
        ----------
        annotation_file: Optional(str|None). We use this file to correct the action name if there was a mistake.

        """
        super().__init__(gpt_model, clip_length = clip_length)
        self.root = root
        self.debug = debug
        self.topk = topk
        self.annotation_file = annotation_file
        self.prediction_file = prediction_file     
        self.handobj_root = handobj_root
        self.question_type = question_type
        self.annotation_root = Path(annotation_file).parent
        self.action_representation = action_representation
        self.labels, self.mapping_vn2narration, self.mapping_vn2act, self.verb_maps, self.noun_maps = generate_label_map(self.annotation_root,                                                                                           
                                                                                            action_representation)                                                                                            
      
        self.gen_type = gen_type
        self.perspective = perspective
        self.benchmark_testing = benchmark_testing
        assert gen_type in ['avion', 'tim', 'random']

        if gen_type == 'avion' or gen_type == 'tim':                  
            self.mc_generator = ActionMultiChoiceGenerator(self.annotation_root)
            assert os.path.exists(self.prediction_file)
            with open(self.prediction_file, 'r') as f:
                self.action_model_predictions = json.load(f)
        else:
            self.mc_generator = RandomMultiChoiceGenerator(self.annotation_root)
            
        self.do_visualization = do_visualization
        self.vis_folder = f"{self.gpt_model}_{self.gen_type}_{self.question_type}_{self.perspective}"
        os.makedirs(self.vis_folder, exist_ok = True)
        self.data = self.init_data()
     
    def save_visualization(self,frames, uid):
        """
        Save the frames to the out_dir
        """
        out_dir = Path(self.vis_folder)
        out_dir.mkdir(parents=True, exist_ok=True)        
        sub_folder = out_dir / uid
        sub_folder.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):            
            cv2.imwrite(str(sub_folder / f"{uid}_{idx}.jpg"), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
     
    def init_data(self):
        ret = {}      
        csv_reader = csv.reader(open(self.annotation_file))
        _ = next(csv_reader) # skip the header

        for idx, row in enumerate(csv_reader):
            narration = row[8]
            pid, vid = row[1:3]
            start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
            vid_path = '{}/{}'.format(pid, vid)
            verb, noun = int(row[10]), int(row[12])
            gt_vn = '{}:{}'.format(verb, noun)            
            narration = row[8]
            
            if self.gen_type == 'avion' or self.gen_type == 'tim':
                action_preds = self.action_model_predictions[str(idx)]['predictions']
                mc_data = self.mc_generator.generate_multi_choice(gt_vn,
                                                            action_preds,
                                                            narration,
                                                            self.topk,
                                                            self.action_representation,
                                                            -1, # n_narrations
                                                            self.labels,
                                                            self.mapping_vn2narration,
                                                            self.verb_maps,
                                                            self.noun_maps,
                                                            benchmark_testing = self.benchmark_testing,
                                                            is_train = False)
            else:
                mc_data = self.mc_generator.generate_multi_choice(gt_vn,
                                                            narration,
                                                            self.topk,
                                                            self.action_representation,
                                                            -1, # n_narrations
                                                            self.labels,
                                                            self.mapping_vn2narration,
                                                            self.verb_maps,
                                                            self.noun_maps,
                                                            benchmark_testing = self.benchmark_testing,
                                                            is_train = False)

            options = mc_data['options'][0]

            ret[idx] = {
                'options': options,
                'gt_answer': narration,
                'start_second': start_second,
                'end_second': end_second,
                'vid_path': vid_path
            }
        return ret

    def multi_process_run(self, offset= 0, n_samples = -1, disable_api_calling = False):
        # inside GPT inference annotator

        if n_samples == -1:
            # do not use offset if n_samples is -1
            assert offset == 0

        if n_samples != -1:
            indices = list(range(len(self.data)))[offset:offset + n_samples]
        else:
            indices = list(range(len(self.data)))
        num_chunks = os.cpu_count() if not self.debug else 2

        indices_groups = self.split_indices(indices, num_chunks)

        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            # Pass additional arguments to the function
            futures = [executor.submit(self.run, group, disable_api_calling) for group in indices_groups]
            
            # Wait for all futures to complete
            combined_results = {}
            for future in futures:
                result_dict = future.result()
                combined_results.update(result_dict)
            print (combined_results)
        if self.debug:
            print (combined_results)
        if combined_results and 'mc_' in self.question_type:
            calculation = calculate_gpt_accuracy(data = combined_results)

        if n_samples == -1:
            n_samples = len(self.data)
        
        checkpoint_name = f"{self.gpt_model}_{self.gen_type}_{self.action_representation}_top{self.topk}_{self.clip_length}f_{n_samples}samples.json"

        if self.do_visualization:
            self.checkpoint(combined_results, os.path.join(self.vis_folder, checkpoint_name))
        self.checkpoint(combined_results, checkpoint_name)                            

    def run(self, indices=None, disable_api_calling = False):      
        if indices is None:
            data_batch = {i : self.data[i] for i in range(len(self.data)) if i in list(range(len(self.data)))}
        else:
            data_batch = {i : self.data[i] for i in range(len(self.data)) if i in indices}
        ret = {}

        for k,v in data_batch.items():
         
            start_timestamp = v['start_second']
            end_timestamp = v['end_second']
            vid_path = v['vid_path']
            _id = v['vid_path'].replace('/', '-')
            uid = f"{_id}_{start_timestamp}_{end_timestamp}"

            frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)
            
            if self.do_visualization:
                # the output folder should reflect the gen type, question type and perspective
                # and the question type
                self.save_visualization(frames, uid)
            if disable_api_calling:
                break
            try:                
                parsed_answer = self.predict_images(frames, v)
            except Exception as e:
                # get full stack trace
                traceback.print_exc()                
                print ("An exception occurred: ", e)
            
            predicted_answer = parsed_answer.answer
            gt_name = v['gt_answer']
            ret[k] = {
                "uid": uid,
                'gt_name': gt_name,
                "options": v['options'],
                'chatgpt_answer': process_raw_pred(predicted_answer) if 'mc_' in self.question_type else predicted_answer
            }
            if self.do_visualization:
                # save ret to the output folder
                self.checkpoint(ret, os.path.join(self.vis_folder, uid, 'inference_results.json'))
            
            if self.debug:
                break
      
        return ret 

    def predict_images(self, images, parsed_item):
        """
        Predict the action from the images
        """
        from llavaction.action.utils import format_task_related_prompt
        options = parsed_item['options']
        start_second = 0
        end_second = parsed_item['end_second'] - parsed_item['start_second']
        temperature = 0
        video_duration = end_second - start_second
        n_frames = len(images)

        task_related_prompt = format_task_related_prompt(options, self.question_type, perspective = self.perspective)

        time_instruction = f"The provided video lasts for {video_duration:.3f} seconds, and {n_frames} frames are uniformly sampled from it. "

        system_prompt = time_instruction + task_related_prompt
        
        format_prompt = """
**Return only a JSON object** with the following two properties:

- `"answer"`: the answer to the question.

"""
     
        if 'o1' in self.gpt_model:
            system_prompt += format_prompt
                    
        if self.handobj_root is not None:
            system_prompt += f"""To further assist you, we mark hands and object when they are visible. The left hand is marked with a bounding box that contains letter L and the right hand's bounding box contains letter R. The object is marked as 'O'."""
        
        if 'o1-mini' == self.gpt_model:
            system_role = "user"
            temperature = 1
        elif 'o1' == self.gpt_model:
            system_role = "developer"
        else:
            system_role = "system"
        
        system_message =  [{"role": system_role, "content": system_prompt}]

        multi_image_content = self.prepare_multiple_images(images)
        multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
        user_message = [{"role": "user", "content": multi_modal_content}]               

        kwargs = {'model': self.gpt_model,
                    'messages': system_message + user_message,
                    'response_format': MultiChoice_Response,
                    'temperature': temperature}
        
        if 'o1' in self.gpt_model:
            kwargs.pop('response_format')
        if 'o1' == self.gpt_model:
            kwargs.pop('temperature')
            pass
            #kwargs['reasoning_effort'] = 'high'
        if 'o1' not in self.gpt_model:
            # structural output
            response = client.beta.chat.completions.parse(
                **kwargs
            )
        else:
            response = client.chat.completions.create(
                **kwargs
            )
            
        total_cost = self.calculate_cost(response)
        
        ret = response.choices[0].message.parsed if 'o1' not in self.gpt_model else response.choices[0].message

        return InferenceAnswer(ret, self.gpt_model)



class GPTHandObjectAnnotator(ChatGPT):
    """
    No need to see the video frames. Just annotate the hand and object
    """
    def __init__(self, ann_file, debug = False):
        super().__init__()
        self.ann_file = ann_file
        self.anno_type = 'gpt-hand-object'
        self.data = []
        self.debug = debug
        with open(ann_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def parse_conversation_from_train_convs(self, item):
        """
        The item has the structure of convs defined in the train anno.
        """
        left_hand_state = item['left_hand_state']
        right_hand_state = item['right_hand_state']
        gt_narration = item['narration']

        ret = {'left_hand_state': left_hand_state,
               'right_hand_state': right_hand_state,
               'gt_narration': gt_narration}
        
        return ret 

    def run(self, indices):

        ret = {}
        for index in indices:
            item = self.data[index]                       
            parsed_item = self.parse_conversation_from_train_convs(item)
            print ('gt_narration', parsed_item['gt_narration'])          
            try:
                gpt_answer = dict(self.annotate(parsed_item))               
            except Exception as e:
                print ("An exception occurred: ", e)
                continue
            conversations = [{'from': 'human', 'value':''}, {'from': 'gpt', 'value': ''}]
            item['conversations'] = conversations                
            item['conversations'][1]['value'] = gpt_answer
            item['question_type'] = self.anno_type
            ret[index] = item
            if self.debug:                
                break

        return ret

    def annotate(self, data_item):
        """
        Assuming that data_item already has the multi-choice options and the gt_answer
        """
        gt_narration = data_item['gt_narration']
        left_hand_state = data_item['left_hand_state']
        right_hand_state = data_item['right_hand_state']
        temperature = 0
        system_prompt = GPTHandObjectPrompt.generate_prompt(left_hand_state, right_hand_state, gt_narration)       
        system_message =  [{"role": "system", "content": system_prompt}]
       
        user_message = [{"role": "user", "content": ""}]

        response = client.beta.chat.completions.parse(
            model=self.gpt_model,
            messages=system_message + user_message, 
            response_format = RESPONSE_FACTORY[self.anno_type],
            temperature = temperature
        )

        total_cost = self.calculate_cost(response)
        ret = response.choices[0].message.parsed        
        return ret 

    def multi_process_run(self, n_samples = -1):
        if n_samples == -1:
            indices = list(range(len(self.data)))
        else:
            indices = list(range(n_samples))[:n_samples]

        sample_suffix = 'all' if n_samples == -1 else str(n_samples)

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

        if self.debug:
            self.checkpoint(combined_results, 'train_anno_debug.json')
        else:
            self.checkpoint(combined_results, f"train_anno_{self.anno_type}_{sample_suffix}.json")
        print ('finished the annotation')
        return combined_results


class GPTAugmentationAnnotator(ChatGPT):
    """
    Given the train annotation from the EK100 dataset, this class will annotate the video frames
    that augments the gt annotations.
    """

    def __init__(self, 
                 ann_file, 
                 root, 
                 clip_length = 4, 
                 debug = False, 
                 anno_type = 'gpt-gt-reason'):
        """
        Parameters
        ----------
        ann_file: jsonl that has the llava's instruction tuning format 
        """
        super().__init__(clip_length = clip_length) 
        self.ann_file = ann_file
        self.root = root
        self.clip_length = clip_length
        self.data = []
        with open(ann_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.debug = debug
        self.anno_type = anno_type

    def parse_conversation_from_train_convs(self, item):
        """
        The item has the structure of convs defined in the train anno.
        """
        conversations = item['conversations']
        human_dict = conversations[0]
        option_text = ', '.join(eval(human_dict['value']))        
        gpt_dict = conversations[1]
        gt_answer = gpt_dict['value']
        assert human_dict['from'] == 'human' and gpt_dict['from'] =='gpt'

        ret = {'options': option_text,
               'gt_answer': gt_answer,
               'start_second': item['start_timestamp'],
               'end_second':  item['end_timestamp']}
        
        return ret

    def multi_process_run(self, n_samples = -1):
        if n_samples == -1:
            indices = list(range(len(self.data)))
        else:
            indices = list(range(n_samples))[:n_samples]

        sample_suffix = 'all' if n_samples == -1 else str(n_samples)

        num_cores = os.cpu_count()  if not self.debug else 2
        indices_groups = self.split_indices(indices, num_cores)
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Pass additional arguments to the function
            futures = [executor.submit(self.run, group) for group in indices_groups]
            
            # Wait for all futures to complete
            combined_results = {}
            for future in futures:
                result_dict = future.result()
                combined_results.update(result_dict)

        if self.debug:
            self.checkpoint(combined_results, 'train_anno_debug.json')
        else:
            self.checkpoint(combined_results, f"train_anno_{self.anno_type}_{self.clip_length}_{sample_suffix}.json")
        print ('finished the annotation')
        return combined_results

    def run(self, indices):
        ret = {}
        for index in indices:
            item = self.data[index]
            start_timestamp = item['start_timestamp']
            end_timestamp = item['end_timestamp']
            vid_path = '{}/{}'.format(item['video'].split('-')[0], item['video'].split('-')[1])
            if self.anno_type in REQUIRES_VIS:
                frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)
            else:
                frames = []
            parsed_item = self.parse_conversation_from_train_convs(item)
            try:
                if self.anno_type == 'gpt-gt-reason' or self.anno_type == 'gpt-gt-strong-reason':
                    gpt_answer = dict(self.annotate(frames, parsed_item))
                elif self.anno_type == 'gpt-gt-instruct-reason':
                    gpt_answer = dict(self.annotate(frames, parsed_item))
            except Exception as e:
                print ("An exception occurred: ", e)
                continue
            
            item['conversations'][1]['value'] = gpt_answer
            item['question_type'] = self.anno_type
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
        start_second = 0
        end_second = data_item['end_second']  - data_item['start_second']
        temperature = 0
        system_prompt = PROMPT_FACTORY[self.anno_type].generate_prompt(start_second, end_second, option_text, gt_answer)

        system_message =  [{"role": "system", "content": system_prompt}]

        if self.anno_type in REQUIRES_VIS:
            multi_image_content = self.prepare_multiple_images(images)
            multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
            user_message = [{"role": "user", "content": multi_modal_content}]
        else:
            user_message = [{"role": "user", "content": ""}]

        response = client.beta.chat.completions.parse(
            model=self.gpt_model,
            messages=system_message + user_message, 
            response_format = RESPONSE_FACTORY[self.anno_type],
            temperature = temperature
        )

        total_cost = self.calculate_cost(response)      

        return response.choices[0].message.parsed
    

def multi_process_annotate(train_file_path, 
                            root, 
                            debug = False, 
                            clip_length = 4,
                            anno_type = 'gpt-gt-reason', 
                            n_samples = -1):
    annotator = GPTAugmentationAnnotator(train_file_path, 
    root, 
    clip_length = clip_length,
    debug = debug,
    anno_type = anno_type)

    annotator.multi_process_run(n_samples = n_samples)

def calculate_gpt_accuracy(path = None, data = None):

    assert path is not None or data is not None
    assert all([path,data]) == False

    if path:
        with open(path, 'r') as f:
            data = json.load(f)

    keys = list(data.keys())
    print ('length of the data', len(keys))

    correct_count = 0
    for k,v in data.items():
        gt_name = v['gt_name']
        chatgpt_answer = v['chatgpt_answer']
        if gt_name == chatgpt_answer:
            correct_count += 1
        else:
            pass
            #print (chatgpt_answer, gt_name)

    print ('accuracy', correct_count / len(keys))

def convert_json_to_jsonl(path):
    with open(path, 'r') as f:
        data = json.load(f)

    disaggree_count = 0
    with open(path.replace('.json', '.jsonl'), 'w') as f:
        for k,v in data.items():
            conversations = v['conversations']
            if isinstance(conversations[1]['value'], dict):
                if 'disagree_with_human_annotation' in conversations[1]['value'] and conversations[1]['value']['disagree_with_human_annotation'] is True:
                    print ('skipping')
                    disaggree_count += 1
                    continue
                new_value = conversations[1]['value']['answer_with_reasoning']
                conversations[1]['value'] = new_value                   
            json.dump(v, f)
            f.write('\n')
    print ('disagree count', disaggree_count)
def calc_disagree_ratio_from_jsonl(path):
    # note it's a jsonl file
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    disagree_count = 0
    for item in data:
        if item['conversations'][1]['value']['disagree_with_human_annotation']:
            print (item)
            disagree_count += 1
    
    print ('disagree ratio', disagree_count / len(data))

def convert_instruct_json_to_jsonl(path, apply_filter = False):
    """
    We split multiple-question answer into multiple lines in the jsonl format. An example of such a json
    "2": {
        "video": "P01-P01_01",
        "conversations": [
            {
                "from": "human",
                "value": "['A. open tap', 'B. pick up knife', 'C. turn off tap', 'D. open drawer', 'E. open cupboard']"
            },
            {
                "from": "gpt",
                "value": {
                    "first_question": "What action is the person performing in the video?",
                    "first_answer": "The person is pulling a drawer open inside a refrigerator.",
                    "second_question": "What evidence suggests that the person is opening a drawer?",
                    "second_answer": "The movement of the drawer outward and the person's hand gripping the handle indicate that the person is opening the drawer.",
                    "third_question": "What is the duration of the action shown in the video?",
                    "third_answer": "The action of opening the drawer is shown in a short duration of 1.230 seconds."
                }
            }
        ],
        "id": "P01-P01_01",
        "split": "train",
        "task_instruction": "",
        "num_samples": 1,
        "question_type": "gpt-gt-instruct-reason",
        "dataset_name": "EK100",
        "start_timestamp": 24.97,
        "end_timestamp": 26.2}
    """
    with open(path, 'r') as f:
        data = json.load(f)
    ret = []
    with open(path.replace('.json', '.jsonl'), 'w') as f:
        for k,v in data.items():
            temp_1 = copy.deepcopy(v)
            temp_2 = copy.deepcopy(v)
            temp_3 = copy.deepcopy(v)
            
            conversations = v['conversations']
            first_question = conversations[1]['value']['first_question']
            first_answer = conversations[1]['value']['first_answer']

            temp_1['conversations'][0]['value'] = first_question
            temp_1['conversations'][1]['value'] = first_answer

            second_question = conversations[1]['value']['second_question']
            second_answer = conversations[1]['value']['second_answer']

            temp_2['conversations'][0]['value'] = second_question
            temp_2['conversations'][1]['value'] = second_answer

            third_question = conversations[1]['value']['third_question']
            third_answer = conversations[1]['value']['third_answer']

            temp_3['conversations'][0]['value'] = third_question
            temp_3['conversations'][1]['value'] = third_answer

            temps = [temp_1, temp_2, temp_3]

            if apply_filter:
                if 'disagree_with_human_annotation' in v['conversations'][1]['value'] and v['conversations'][1]['value']['disagree_with_human_annotation'] is True:
                    continue   
                random_index = np.random.randint(0, 3)               
                ret.append(temps[random_index])
            else:
                ret.append(temp_1)
                ret.append(temp_2)
                ret.append(temp_3)
             
        for item in ret:
            json.dump(item, f)
            f.write('\n')
    

if __name__ == '__main__':    

    root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    val_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
    avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'    

    annotator = GPTInferenceAnnotator(root, 
    val_file,
    avion_prediction_file,
    clip_length = 8,
    debug = False,
    action_representation = "GT_random_narration",
    question_type = 'mc_GT_random_narration',
    topk = 5)  

    annotator.multi_process_run(n_samples = 100)
    print ('# json errors', InferenceAnswer.json_errors)