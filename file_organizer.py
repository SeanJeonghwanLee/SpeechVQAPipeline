import os
import json
from tqdm import tqdm
from collections import defaultdict

def anno_maker(root_path:str):
    # merging trainset only
    # train_dict = {"annotations" : []}
    train_item = []
    train_path = f"./train_labels"
    categories = os.listdir(train_path)
    for category in categories:
        for lvl in ['상','중','하']: # very last dir
            path = f"{train_path}/{category}/{lvl}_train_{category}"
            with open(f"{path}/annotation.json","r") as anno_f:
                anno_json = json.loads(anno_f)
            with open(f"{path}/images.json","r") as img_f:
                img_json = json.loads(img_f)
            with open(f"{path}/question.json","r") as que_f:
                que_json = json.loads(que_f)
            
            # required data
            annotations = anno_json["annotations"]
            images = img_json["images"]
            questions = que_json["questions"]
            
            # image dictionary : to be used when merging them all together, image path extraction
            _img_dict = {}
            for img_set in images:
                assert img_set["image_id"] not in _img_dict
                _img_dict[img_set["image_id"]] = img_set["image"]
            # annotation dictionary : to be used when merging them all together, answer extraction
            _anno_dict = {}
            for anno_set in annotations:
                assert anno_set["question_id"] not in _anno_dict
                _anno_dict[anno_set["question_id"]] = anno_set["multiple_choice_answer"]               
            # question dictionary : the base dictionary, image path and answer will be merged here
            for que_set in questions:
                _temp_que_dict = que_set
                _temp_que_dict["answer"] = _anno_dict[que_set["question_id"]]
                _temp_que_dict["image"] = _anno_dict[que_set["image_id"]]
                train_item = 
                
                
                
                
            
    # with open(f"{root_path}/train.json", 'w') as train_json:
    
    
    
    # splitting into train/val/test
    # putting trainset all together
    