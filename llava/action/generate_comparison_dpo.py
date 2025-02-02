## first inference the dataset with one frame only
## cannot be the whole dataset but a subset of it
## then inference the dataset with 8 frames
## Then find where the one frame is wrong and 8 frames is right
## collect those examples as data for dpo


from llava.action.chatgpt_utils import ChatGPT
import os
import csv
import random
from tqdm import tqdm
from pydantic import BaseModel
import traceback
from concurrent.futures import ProcessPoolExecutor
import openai


client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


GPT_MODEL = 'gpt-4o'

class CaptionResponse(BaseModel):
    """
    The GT was known. The response is to add more information to the GT
    """
    caption: str


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

class CaptionInference(ChatGPT):
    def __init__(self, 
                 root,                 
                 annotation_file,
                 clip_length = 4,
                 debug = False
                 ):    
        self.root = root
        self.annotation_file = annotation_file
        self.clip_length = clip_length        
        self.debug = debug
        self.question_type = 'gpt-gt-reason'        
                
        self.data = self.init_data()        
        
        print (len(self.data))

    def select_train_subset(self):
        
        with open(os.path.join(self.annotation_file), 'r') as f:    
            csv_reader = list(csv.reader(f))
        header = csv_reader[0]  # Get header
        data = csv_reader[1:]   # Get data
        N = len(data)
        print ('N', N)
        # get a random subset of the data such as 20% of them. Give the indices
        random.seed(0)
        indices = random.sample(range(N), int(N*0.2))
        return indices
        
    def init_data(self):
        ret = {}      
        csv_reader = csv.reader(open(self.annotation_file))
        _ = next(csv_reader) # skip the header

        indices = self.select_train_subset()
        count = 0
        for idx, row in enumerate(csv_reader):
            if idx not in indices:
                continue 
            narration = row[8]
            pid, vid = row[1:3]
            start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
            vid_path = '{}/{}'.format(pid, vid)
            verb, noun = int(row[10]), int(row[12])
            gt_vn = '{}:{}'.format(verb, noun)

            narration = row[8]           

            ret[count] = {
                'gt_answer': narration,
                'start_second': start_second,
                'end_second': end_second,
              
                'vid_path': vid_path
            }
            count+=1
        return ret  
    
    def multi_process_run(self, n_samples = -1):
        # to initialize it

        if n_samples != -1:
            indices = list(range(len(self.data)))[:n_samples]

        num_chunks = os.cpu_count() if not self.debug else 1

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

    
    def predict_images(self, images, parsed_item):
        """
        Predict the action from the images
        """
        from llava.action.utils import format_task_related_prompt
        options = parsed_item['options']
        start_second = 0
        end_second = parsed_item['end_second'] - parsed_item['start_second']
        temperature = 0
        video_duration = end_second - start_second
        n_frames = len(images)

        task_related_prompt = format_task_related_prompt(options, self.question_type, perspective = 'first_person')

        time_instruction = f"The provided video lasts for {video_duration:.3f} seconds. "

        system_prompt = time_instruction + task_related_prompt
        
        format_prompt = """
**Return only a JSON object** with the following two properties:

- `"answer"`: the answer to the question.
- `"caption"`: A detailed caption of the video. Used to support the answer.
"""
     
        if 'o1' in GPT_MODEL:
            system_prompt += format_prompt
     
        print (system_prompt)
              
        if 'o1-mini' == GPT_MODEL:
            system_role = "user"
            temperature = 1
        elif 'o1' == GPT_MODEL:
            system_role = "developer"
        else:
            system_role = "system"
        
        system_message =  [{"role": system_role, "content": system_prompt}]

        multi_image_content = self.prepare_multiple_images(images)
        multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
        user_message = [{"role": "user", "content": multi_modal_content}]               

        kwargs = {'model': GPT_MODEL,
                    'messages': system_message + user_message,
                    'response_format': CaptionResponse,
                    'temperature': temperature}
        
        if 'o1' in GPT_MODEL:
            kwargs.pop('response_format')
        if 'o1' == GPT_MODEL:
            kwargs.pop('temperature')
            pass
            #kwargs['reasoning_effort'] = 'high'
        if 'o1' not in GPT_MODEL:
            # structural output
            response = client.beta.chat.completions.parse(
                **kwargs
            )
        else:
            response = client.chat.completions.create(
                **kwargs
            )
            
        total_cost = self.calculate_cost(response)
        
        ret = response.choices[0].message.parsed if 'o1' not in GPT_MODEL else response.choices[0].message

        return ret
    
    def run(self, indices = None):
                             
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
                # get full stack trace
                traceback.print_exc()
                
                print ("An exception occurred: ", e)

            caption = parsed_answer.caption
            print ('caption:', caption)

            if self.debug:
                break
       
        return ret

                  




if __name__ == '__main__':
    video_root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    anno_root = '/data/shaokai/epic-kitchens-100-annotations/'
    clip_length = 8
        
    cap = CaptionInference(video_root, os.path.join(anno_root, 'EPIC_100_train.csv'), clip_length, debug = True)
    
    #cap.multi_process_run(n_samples = 2)
    cap.run()