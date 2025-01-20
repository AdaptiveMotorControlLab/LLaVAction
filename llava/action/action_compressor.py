from llava.action.chatgpt_utils import ChatGPT
import json
import openai
import os
from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel



client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class CompressedNarration(BaseModel):  
    compressed_narration: str

class ActionCompressor(ChatGPT):
    def __init__(self, vnstr2narration_path, debug=False):
        self.vnstr2narration = json.load(open(vnstr2narration_path))        
        # deduplicate the narrations
        for vn_str, narration_list in self.vnstr2narration.items():
            self.vnstr2narration[vn_str] = list(set(narration_list))
        self.debug = debug
        self.keys = list(self.vnstr2narration.keys())
        
    def map(self, vn_str, narration_list):
        
        instruction = """
You are given a list of descriptions of actions. Please compress the list into a short sentence.
TIPS: ALl these descriptions involve nouns and verbs. In your compressed version, try to use nouns and verbs that can cover all the descriptions.
If you can't use one single noun or verbs to cover all the descriptions, you can use multiple nouns or verbs. But be sure to be precise. Don't be too general. It's better to be verbose than being vague.
Note your answer should only include a short sentence without an ending period.

Example compressed_narration: "cutting or slicing the vegetables"
"""   
        system_role = "system"
        system_message =  [{"role": system_role, "content": instruction}]
        
        narration_list = ", ".join(narration_list)
        user_message = [{"role": "user", "content": [{"type": "text", "text": f"narration_list: {narration_list}"}]}] 
        
        kwargs = {'model': 'gpt-4o',
                  'response_format': CompressedNarration,
                  'messages': system_message + user_message}                     

        response = client.beta.chat.completions.parse(
                **kwargs
            )
        result = response.choices[0].message.parsed.compressed_narration

        return result
    def run(self, indices):

        ret = {}
        for index in indices:
            key = self.keys[index]
            narration_list = self.vnstr2narration[key]
            new_action_str = self.map(key, narration_list)
            ret[key] = {'new_gt': new_action_str,
                        'gt_narrations': narration_list} 

        return ret              
        
    def multi_run(self, n_samples = -1):                        
        if n_samples == -1:
            indices = list(range(len(self.vnstr2narration)))
        else:
            indices = list(range(n_samples))
        num_chunks = os.cpu_count() * 2 if not self.debug else 2
        indices_groups = self.split_indices(indices, num_chunks)        
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            
            futures = [executor.submit(self.run, group) for group in indices_groups]
            
            # Wait for all futures to complete
            combined_results = {}
            for future in futures:
                result_dict = future.result()
                combined_results.update(result_dict)
        
        print ('combined results')
        print (len(combined_results))
        with open('mapping_vnstr2narration_gpt.json', 'w') as f:
            json.dump(combined_results, f, indent=4)
        
        
if __name__ == '__main__':

    compressor = ActionCompressor('mapping_vnstr2narration.json')

    compressor.multi_run(n_samples = -1)

