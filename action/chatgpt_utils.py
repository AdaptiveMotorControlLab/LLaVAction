import base64
import io
import json
import os
import cv2
import numpy as np
import openai
from pydantic import BaseModel
from multiprocessing.pool import Pool

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



class GPTAnnotator:
    def __init__(self, prediction_file_path):
        with open(prediction_file_path, 'r') as f:
            self.prediction_file = json.load(f)

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


    def annotate(self, images):
        """
        Annotate to do image caption only
        """
        pass

    def annotate_with_multichoice(self, images, mc_data):
        """
        Annotate with mc_data

        {

        }

        """

        temperature = 0
        include_images = True

        system_prompt_prefix = """Inspect the images from the video and explain why the answer of the multi-choice question is D. """
        system_prompt_suffix = """Yes"""

        system_prompt = system_prompt_prefix + system_prompt_suffix

        system_message =  [{"role": "system", "content": system_prompt}]

        if include_images:
            multi_image_content = self.prepare_multiple_images(images)
            multi_modal_content = [{"type": "text", "text": ""}] + multi_image_content
            user_message = [{"role": "user", "content": multi_modal_content}]
        else:
            user_message = [{"role": "user", "content": ""}]        

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
    
def annotate_from_train_conv_file(train_file_path):
    pass

if __name__ == '__main__':
    train_file_path = '/storage-rcp-pure/upmwmathis_scratch/shaokai'
    annotate_from_train_conv_file(train_file_path)
