import os
import sys
sys.path[0] = os.path.dirname(os.path.dirname(sys.path[0]))
import openai
from pydantic import BaseModel
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from llava.action.utils import AvionMultiChoiceGenerator
from llava.action.utils import avion_video_loader, avion_video_render_loader
import copy 
import torch
import io
import numpy as np 
import base64

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-4o-2024-08-06"

prices = {
    "gpt-4o-2024-08-06": {"input": 2.5 / 10**6, "output": 10 / 10**6},
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

        prompt = f"""Your job is to create 3 question and answer pairs based on the text below.
        {reason_mc_string}
        Example questions you can ask include. Note you are not limited to these questions:
        What object the person is interacting with?
        What objects are visible in the video?
        What is the sequence of the atomic actions that the person is performing?
        Make sure your only ask questions that can be answered with enough grounding in the text.

        """
        return prompt


class LLaVAWrongAnswerAwarePrompt:
    """
    The prompt for the annotation
    """
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):
        prompt = f"""
You are seeing video frames from an egocentric view of a person. 
Please talk as if you are the person in the video and describe what action you are performing.
To assist you for how to describe the action, the video's start time is {start_second} and the end time is {end_second} and the duration is {end_second - start_second} seconds.

To further assist you for how to describe the action, note that in a multi-choice video question answering, you were given following options {option_text} and the correct answer is {gt_answer}.
In addition to describe what you see, describe why wrong answers were wrong and why right answer was right.
When you explain why wrong answers were wrong and why right answer was right, you should use the following flow of reasoning:

The flow of reasoning:
1. What objects need to be visible to support the answer?
2. Whether the duration in time supports that answer?

Based on the answers above, why right answer is right and why wrong answers were wrong."""
        return prompt   


class GPTReasoningWithoutGTPrompt:
    """
    The perhaps simplest reasoning explanation.
    """
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):
        prompt = f"""
You are seeing video frames from an egocentric view of a person. The person is interacting with objects in a kitchen.
Describe the action the person is performing. Pay attention to the objects the person's hands are interacting.
Explain in details what are the supporting evidences for the action. Useful evidences include the duration of the video, the objects the person is interacting with, and the context of the video. 
"""
        return prompt

class GPTReasoningWithGTPrompt:
    @classmethod
    def generate_prompt(cls, start_second, end_second, option_text, gt_answer):
        prompt = f"""
You are viewing video frames from an egocentric perspective of a person interacting with objects in a kitchen. Describe the video frames in detail and reason about the actions the person is performing. You will be provided with the human-annotated ground-truth for the action, but you should independently come to your own conclusion.
If you disagree with the human annotation, indicate "true" in the "disagree_with_human_annotation" field of your response, and provide your reasoning without mentioning the ground-truth answer. This will keep your reasoning clean. If you agree with the human annotation, indicate "false" in the "disagree_with_human_annotation" field and provide your reasoning without referencing the ground-truth to maintain a clean description.
Pay close attention to the objects the person's hands are interacting with.
The true ground-truth action is {gt_answer}.
Your reasoning steps should include supporting evidence for the action, such as the duration of the video, the sequence of actions the person performs, the objects they interact with, and the overall context of the video.
As a general guideline, for videos longer than 3 seconds, provide detailed reasoning steps, and for videos shorter than 3 seconds, generate less detailed reasoning.
The video duration is {end_second - start_second:.3f} seconds.
"""
        print (prompt)
        return prompt    

class GT_Augmentation_Response(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    caption_with_reasoning: str
    disagree_with_human_annotation: bool


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
                   'gpt-gt-instruct-reason': ExpandReasonMCPrompt}

REQUIRES_VIS = set(['gpt-gt-reason'])

RESPONSE_FACTORY = {'gpt-gt-reason': GT_Augmentation_Response,
                    'gpt-gt-instruct-reason': ExpandReasonMCResponse}

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

    def extract_frames(self,  vid_path, start_second, end_second):
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
                 root,                 
                 annotation_file,
                 avion_prediction_file,
                 handobj_root = None,
                 clip_length = 4, 
                 action_representation = 'GT_random_narration',
                 debug = False,
                 topk = 10,
                 ):
        """
        Parameters
        ----------
        annotation_file: Optional(str|None). We use this file to correct the action name if there was a mistake.

        """
        super().__init__(clip_length = clip_length)
        self.root = root
        self.debug = debug
        self.topk = topk
        self.annotation_file = annotation_file
        self.avion_prediction_file = avion_prediction_file     
        self.handobj_root = handobj_root
        self.annotation_root = Path(annotation_file).parent
        self.action_representation = action_representation
        self.labels, self.mapping_vn2narration, self.mapping_vn2act, self.verb_maps, self.noun_maps = generate_label_map(self.annotation_root,                                                                                           
                                                                                            action_representation,
                                                                                            cache_file =  os.path.join(self.annotation_root, 'nlp_cache.pkl'))

      
        self.mc_generator = AvionMultiChoiceGenerator(self.annotation_root)
        with open(avion_prediction_file, 'r') as f:
            self.avion_predictions = json.load(f)

        self.data = self.init_data()
       
     
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
            avion_preds = self.avion_predictions[str(idx)]['predictions']
            narration = row[8]
            mc_data = self.mc_generator.generate_multi_choice(gt_vn,
                                                        avion_preds,
                                                        narration,
                                                        self.topk,
                                                        self.action_representation,
                                                        -1, # n_narrations
                                                        self.labels,
                                                        self.mapping_vn2narration,
                                                        self.verb_maps,
                                                        self.noun_maps,
                                                        is_train = False)

            options = mc_data['options'][0]

            option_string = ','.join(options)
            ret[idx] = {
                'options': option_string,
                'gt_answer': narration,
                'start_second': start_second,
                'end_second': end_second,
                'vid_path': vid_path
            }

        return ret

    def multi_process_run(self, n_samples = -1):
        # to initialize it

        if n_samples != -1:
            indices = list(range(len(self.data)))[:n_samples]

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

        if self.debug:
            print (combined_results)

        calculation = calculate_gpt_accuracy(data = combined_results)

        self.checkpoint(combined_results, "gpt_inference_results.json")                            

    def run(self, indices=None):
        if indices is None:
            data_batch = {i : self.data[i] for i in range(len(self.data)) if i in list(range(len(self.data)))}
        else:
            data_batch = {i : self.data[i] for i in range(len(self.data)) if i in indices}
        ret = {}

        for k,v in tqdm(data_batch.items()):            
         
            start_timestamp = v['start_second']
            end_timestamp = v['end_second']
            vid_path = v['vid_path']

            frames, time_meta = self.extract_frames(vid_path, start_timestamp, end_timestamp)
            try:
                parsed_answer = self.predict_images(frames, v)
            except Exception as e:
                print ("An exception occurred: ", e)
            predicted_answer = parsed_answer.answer
            explanation = parsed_answer.explanation
            print (explanation)
            gt_name = v['gt_answer']
            ret[k] = {
                'gt_name': gt_name,
                'chatgpt_answer': predicted_answer,
                'explanation': explanation
            }
            if self.debug:
                break
        if indices is None:
            calculation = calculate_gpt_accuracy(data = ret)
            self.checkpoint(ret, "gpt_inference_results.json")
        else:
            return ret 



    def predict_images(self, images, parsed_item):
        """
        Predict the action from the images
        """        
        option_text = parsed_item['options']
        start_second = 0
        end_second = parsed_item['end_second'] - parsed_item['start_second']
        temperature = 0
        duration = end_second - start_second

        system_prompt = f"""
        You are seeing video frames from an egocentric view of a person. Pretend that you are the person.  Your task is to describe what action you are performing.
        To assist you for how to describe the action, the video's start time is {start_second} and the end time is {end_second:.3f} and the duration is {duration:.3f} seconds.
        You were given multiple choice options {option_text}. Pick the correct one and put that into the answer. Note in the answer do not include the option letter, just the name of the action.        
        """

        if self.handobj_root is not None:
            system_prompt += f"""To further assist you, we mark hands and object when they are visible. The left hand is marked with a bounding box that contains letter L and the right hand's bounding box contains letter R. The object is marked as 'O'."""
        
        system_prompt += f"""Before giving the answer, explain why the correct answer is correct and why the other options are incorrect. You must pay attention to the hands and objects to support your reasoning when they are present."""


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
        option_text = ','.join(eval(human_dict['value']))        
        gpt_dict = conversations[1]
        gt_answer = gpt_dict['value']
        print ('gt_answer', gt_answer)
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

        num_cores = os.cpu_count() * 2 if not self.debug else 2
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
        for index in tqdm(indices):
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
                if self.anno_type == 'gpt-gt-reason':
                    gpt_answer = dict(self.annotate(frames, parsed_item))
                elif self.anno_type == 'gpt-gt-instruct-reason':
                    gpt_answer = dict(self.annotate(frames, parsed_item))
            except Exception as e:
                print ("An exception occurred: ", e)
                continue
            
            item['conversations'][1]['value'] = gpt_answer
            item['question_type'] = self.anno_type
            ret[index] = item
            print (item)
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
            model=GPT_MODEL,
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

def multi_process_inference(root,
                            annotation_file, 
                            avion_prediction_file,
                            handobj_root = None,
                            action_representation = 'GT_random_narration',
                            clip_length = 4,
                            topk = 5,                             
                            debug = False,
                            n_samples = -1
                            ):

    annotator = GPTInferenceAnnotator(root, 
    annotation_file,
    avion_prediction_file,
    handobj_root = handobj_root,
    clip_length = clip_length,
    debug = debug,
    action_representation = action_representation,
    topk = topk)

    # indices = list(range(len(annotator.data)))[:100]
    # annotator.run()

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
            print (chatgpt_answer, gt_name)

    print ('accuracy', correct_count / len(keys))

def convert_json_to_jsonl(path):
    with open(path, 'r') as f:
        data = json.load(f)

    with open(path.replace('.json', '.jsonl'), 'w') as f:
        for k,v in data.items():
            json.dump(v, f)
            f.write('\n')
def convert_instruct_json_to_jsonl(path):
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

            ret.append(temp_1)
            ret.append(temp_2)
            ret.append(temp_3)
             
        for item in ret:
            json.dump(item, f)
            f.write('\n')
    

if __name__ == '__main__':    

    # amg0
    # train_file_path = '/data/epic_kitchen/AVION_PREDS/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'
    # root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    # val_file = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv'
    # avion_prediction_file = '/data/epic_kitchen/AVION_PREDS/avion_pred_ids_val.json'
    # handobj_root = '/data/epic_kitchen/Save_dir'

    # haozhe's path
    # root = '/mediaPFM/data/haozhe/onevision/llava_video/EK100'
    # val_file = '/mediaPFM/data/haozhe/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'
    # avion_prediction_file = '/mediaPFM/data/haozhe/EK100/EK100_in_LLAVA/avion_pred_ids_val.json'
    # handobj_root = '/mnt/SV_storage/VFM/hand_object_detector/Save_dir'


    #root = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100'
    #train_file_path = '/storage-rcp-pure/upmwmathis_scratch/shaokai/AVION_PREDS/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'

    #train_file_path = '/data/epic_kitchen/shaokai_explore/LLaVA-NeXT/train_anno_gpt-gt-reason_4_all.jsonl'
    train_file_path = '/data/epic_kitchen/AVION_PREDS/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'
    root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    multi_process_annotate(train_file_path, 
                    root, 
                    debug = True, 
                    clip_length = 8,
                    n_samples = -1, anno_type = 'gpt-gt-reason')

    # multi_process_inference(root, 
    #                         val_file, 
    #                         avion_prediction_file,
    #                         handobj_root = handobj_root,
    #                         debug = False,
    #                         clip_length = 8,
    #                         topk = 5,
    #                         n_samples = 100)


    # convert_json_to_jsonl('train_anno_gpt-gt-reason_4_10000.json')

    #convert_instruct_json_to_jsonl('train_anno_gpt-gt-instruct-reason_4_all.json')