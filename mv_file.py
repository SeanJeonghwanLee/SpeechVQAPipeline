import os
from tqdm import tqdm

mode_list = ['mv', 'unzip', 'anno']

mode = 'anno'


root_path = '/home/seanlee/class/deeplearning/ko_dataset/validation'
folder_list = os.listdir(f"{root_path}/images")
anno_dict = {}

for folder in tqdm(folder_list):
    sub_root_path = f"{root_path}/images/{folder}"
    if mode == 'mv':
        for x in ['상','중','하']:
            print(f'for f in {sub_root_path}/{x}_validate_{folder}/*; do mv "$f" {root_path}/; done')
            os.system(f'for f in {sub_root_path}/{x}_validate_{folder}/*; do mv "$f" {root_path}/; done')
    elif mode == 'unzip':
        print(f'unzip -qq {sub_root_path}/{folder}.zip -d {sub_root_path}/')
        os.system(f'unzip -qq {sub_root_path}/{folder}.zip -d {sub_root_path}/')
    elif mode == 'anno':
        anno_sub_dict = {}
        for x in ['상','중','하']: #image_path, question, question_id, answer
            anno_json = json.loads(f"{sub_root_path}/{x}_validate_{folder}/annotation.json")
            images_json = json.loads(f"{sub_root_path}/{x}_validate_{folder}/annotation.json")
            anno_json = json.loads(f"{sub_root_path}/{x}_validate_{folder}/annotation.json")
            
            for anno
    elif mode not in mode_list:
        print(f"####Invalid Mode####")