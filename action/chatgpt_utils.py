import base64
import io
import json
import os
import numpy as np
import openai
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from action.utils import avion_video_loader
import torch
import cv2
from pathlib import Path

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
    # Calculate the size of each chunk and the remainder
    chunk_size = len(indices) // num_chunks
    remainder = len(indices) % num_chunks

    # Create chunks, distributing the remainder across the first few chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Each of the first 'remainder' chunks will have one extra element
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(indices[start:end])
        start = end

    return chunks

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


    def extract_frames(self, data_root, vid_path, start_second, end_second):
        frames, time_meta = avion_video_loader(data_root,
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

    def parse_conversation(self, item):
        """
        We should get time steps, duration
        We shoudd also get gt and wrong answers
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

    def annotate(self, indices):

        data_batch = [self.data[i] for i in range(len(self.data)) if i in indices]

        ret = {}
        for index in indices:
            item = self.data[index]
            start_timestamp = item['start_timestamp']
            end_timestamp = item['end_timestamp']
            vid_path = '{}/{}'.format(item['video'].split('-')[0], item['video'].split('-')[1])
            frames, time_meta = self.extract_frames(self.data_root, vid_path, start_timestamp, end_timestamp)
            parsed_item = self.parse_conversation(item)
            gpt_answer = self.annotate_images(frames, parsed_item).explanation
            item['conversations'][1]['value'] = gpt_answer
            ret[index] = item
            break

        return ret         

    def annotate_images(self, images, data_item):
        """
        Annotate with mc_data
        {
        }
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
            response_format = MultiChoiceResponse,
            temperature = temperature
        )

        return response.choices[0].message.parsed
    

def process_subset(indices_subset, train_file_path, root):
    # Initialize a new annotator instance within each process
    annotator = GPTAnnotator(train_file_path, root)
    return annotator.annotate(indices_subset)


if __name__ == '__main__':
    #train_file_path = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train/avion_mc_top10/train_convs_narration.jsonl'
    #root = '/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100'
    train_file_path = '/data/EK100_inst_train/avion_mc_top10/train_convs_narration.jsonl'
    root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    
    num_cores = 2 #os.cpu_count()

    print (f'Using {num_cores} cores thus splitting the data into {num_cores} chunks')

    with open(train_file_path, 'r') as f:
        num_lines = len([line for line in f])

    print (f'Total number of lines in the file: {num_lines}')
    indices = list(range(num_lines))
    print ('indices', len(indices))
    
    indices_groups = split_indices(indices, num_cores)

    print ('number of groups')
    print (len(indices_groups))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Pass additional arguments to the function
        futures = [executor.submit(process_subset, group, train_file_path, root) for group in indices_groups]
        
        # Wait for all futures to complete
        combined_results = {}
        for future in futures:
            result_dict = future.result()
            combined_results.update(result_dict)
        
    keys = sorted(list(combined_results.keys()))

    print ('resulted number of keys', len(keys))
    
    result = []
    for key in keys:
        result.append(combined_results[key])

    anno_root = Path(train_file_path).parent

    with open(anno_root / 'gpt_annotated.jsonl', 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')