import base64
import io
import json
import os
import numpy as np
import openai
from pydantic import BaseModel
from multiprocessing.pool import Pool
from action.utils import avion_video_loader
import cv2

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-4o-2024-08-06"


class ImageOnlyResponse(BaseModel):
    """
    """
    explanation: str

class MultiChoiceResponse(BaseModel):
    """
    The output format of the response
    """

    explanation: str


def split_indices(indices, num_chunks):
    chunk_size = len(indices) // num_chunks
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

class GPTAnnotator:
    def __init__(self, ann_file, data_root, clip_length = 32):
        self.ann_file = ann_file
        self.data_root = data_root
        self.clip_length = clip_length
        data = []
        with open(ann_file, 'r') as f:
            for line in f:
            # Parse the JSON data
                _data = json.loads(line)
                # Process your data
                data.append(_data)
        self.data = data
        

    def prepare_multiple_images(self, images):
        """

        """               
        encoded_image_list = []

        for image in images:
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


    def extract_frames(self, data_root, vid_path, start_second, end_second):
        frames, time_meta = avion_video_loader(data_root,
                        vid_path,
                        'MP4',
                        start_second,
                        end_second,
                        clip_length = self.clip_length,
                        threads = 1,
                        fast_rrc=False,
                        fast_rcc = False,
                        jitter = False)
        return frames, time_meta

    def parse_conversation(self, item):
        """
        We should get time steps, duration
        We shoudd also get gt and wrong answers
        """
        conversations = item['conversations']
        human_dict = conversations[0]

        # the offset is to remove the quote ' 
        option_start = human_dict['value'].index['['] + 2
        option_end = human_dict['value'].index[']'] - 1

        option_text =  human_dict['value'][option_start:option_end]        
        gpt_dict = conversations[1]
        gt_answer = gpt_dict['value']

        assert human_dict['from'] == 'human' and gpt_dict['from'] =='gpt'

        ret = {'options': option_text,
               'gt_answer': gt_answer,
               'start_second': item['start_timestamp'],
               'end_second':  item['end_timestemp']}
        
        return ret

    def annotate(self, indices):

        data_batch = [self.data[i] for i in range(len(self.data)) if i in indices]

        for item in data_batch:
            start_timestamp = item['start_timestamp']
            end_timestamp = item['end_timestamp']
            vid_path = '{}/{}'.format(item['video'].split('-')[0], item['video'].split('-')[1])
            frames, time_meta = self.extract_frames(self.data_root, vid_path, start_timestamp, end_timestamp)
            data_item = self.parse_conversation(item)
            anno = self.annotate_images(frames, data_item)
            print (anno)
            break

    def annotate_images(self, images, data_item):
        """
        Annotate with mc_data
        {
        }
        """
        gt_answer = data_item['gt_answer']
        option_text = data_item['option_text']
        start_second = data_item['start_second']
        end_second = data_item['end_second']        
        temperature = 0
        system_prompt_prefix = f"""
You are seeing video frames from an egocentric view. You are determining what action the person is performing.
The video's start time is {start_second} and the end time is {end_second}.
In a multi-choice video question answering, you were given following options {option_text} and the correct answer is {gt_answer}.
Please describe what you see and why wrong answers are wrong and why right answer is right.
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
            response_format = MultiChoiceResponse,
            temperature = temperature
        )

        return response.choices[0].message.parsed
    

def annotate_using_chatgpt():
    """
    Multi processing to speed up 
    """
    with Pool() as pool:
        pass
        #pool.starmap(annotate, task_args)

    pass



if __name__ == '__main__':
    train_file_path = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train/avion_mc_top10/train_convs_narration.jsonl'
    root = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100'

    
    GPTAnnotator(train_file_path, root)