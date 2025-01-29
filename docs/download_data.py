import os
os.environ["HF_HOME"] = "/mnt/SV_storage/VFM/huggingface"
from datasets import load_dataset
from datasets import get_dataset_config_names, get_dataset_split_names
from tqdm import tqdm
import json
import yaml

dataset_name = "lmms-lab/LLaVA-Video-178K"

save_root = "/mnt/SV_storage/VFM/onevision/llava_video_178k"

subsets = get_dataset_config_names(dataset_name)
for subset in subsets:
    # download the dataset
    data = load_dataset(dataset_name, subset)
    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        json_data["video"] = da["video"]
        json_data["conversations"] = da["conversations"]
        with open(os.path.join(save_root, '{}.json'.format(da["id"])), "w") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    aa= 1

    # splits = get_dataset_split_names(dataset_name, subset)
    
        
#     aa = 1


# data = load_dataset("lmms-lab/LLaVA-Video-178K", '0_30_s_academic_v0_1', split="caption")

# for da in tqdm(data):
#     json_data = {}
#     json_data["id"] = da["id"]
#     aa= 2

avaliable_datasets = ['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 
                      'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 
                      'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)', 
                      'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 
                      'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 
                      'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 
                      'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 
                      'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 
                      'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 
                      'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 
                      'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 
                      'intergps(cauldron,llava_format)', 'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 
                      'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 
                      'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 
                      'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 
                      'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 
                      'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 
                      'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 
                      'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 
                      'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 
                      'tqa(cauldron,llava_format)', 'ureader_cap', 'ureader_ie', 'vision_flan(filtered)', 
                      'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 
                      'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']

# chossen_datasets = ['sharegpt4v(sam)', 'sharegpt4v(llava)']
chossen_datasets = ['geo3k']

image_base = "/mnt/SV_storage/VFM/onevision/llava_data"
json_base = "/mnt/SV_storage/VFM/onevision/llava_instruct"
# dataset_yaml = 'scripts/train/onevision.yaml'

# # open the yaml file
# with open(dataset_yaml, 'r') as f:
#     dataset_config = yaml.safe_load(f)

# dataset_paths = {}
# for data_info in dataset_config['datasets']:
#     dataset_paths[data_info['json_path'].split('/')[-1]] = data_info['json_path']


for dataset_name in chossen_datasets:
    data = load_dataset("lmms-lab/LLaVA-OneVision-Data", dataset_name, split="train")
    converted_data = []

    image_folder = os.path.join(image_base, dataset_name)
    os.makedirs(image_folder, exist_ok=True)

    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image"] = f"{da['id']}.png"
            da["image"].save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


    with open(os.path.join(json_base, '{}.json'.format(dataset_name)), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)