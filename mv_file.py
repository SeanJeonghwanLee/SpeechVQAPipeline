import os
import os.path
from tqdm import tqdm
import json
from collections import deque


mode_list = ['mv', 'unzip', 'check']

mode = 'check'


root_path = '/home/seanlee/class/deeplearning/ko_dataset/validation'
anno_dict = {}



if mode == 'check':
    existing_file = deque()
    querying_file = deque()
    with os.scandir("/home/seanlee/class/SpeechVQAPipeline/images") as xx:
        for x in tqdm(xx):
            existing_file.append(x.name)

    for split in tqdm(['train','val','test']):
        json_lst = deque()
        with open(f"/home/seanlee/class/SpeechVQAPipeline/vqa.{split}.json", 'r') as ff:
            for f in tqdm(ff):
                json_lst.append(json.loads(f))
        
        for vqa_train_json in tqdm(json_lst):
            querying_file.append(vqa_train_json["image_path"])
    non_existing_file = set(querying_file) - set(existing_file)
    print(f"len(non_existing_file):{len(non_existing_file)}")
    with open(f"/home/seanlee/class/SpeechVQAPipeline/{split}_non_existing.txt", 'w') as f:
        for k in non_existing_file:
            f.write(f"{k}\n")
else:
    folder_list = os.listdir(f"{root_path}/images")
    for folder in tqdm(folder_list):
        sub_root_path = f"{root_path}/images/{folder}"
        if mode == 'mv':
            for x in ['상','중','하']:
                print(f'for f in {sub_root_path}/{x}_validate_{folder}/*; do mv "$f" {root_path}/; done')
                os.system(f'for f in {sub_root_path}/{x}_validate_{folder}/*; do mv "$f" {root_path}/; done')
        elif mode == 'unzip':
            print(f'unzip -qq {sub_root_path}/{folder}.zip -d {sub_root_path}/')
            os.system(f'unzip -qq {sub_root_path}/{folder}.zip -d {sub_root_path}/')

        elif mode not in mode_list:
            print(f"####Invalid Mode####")