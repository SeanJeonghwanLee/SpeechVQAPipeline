import os
import json
from tqdm import tqdm
from collections import defaultdict



img_dup = {}
que_id_dup = {}

def anno_maker():
    # merging trainset only
    train_only_item = []
    train_path = f"./train_labels"
    categories = os.listdir(train_path)
    for category in tqdm(categories):
        for lvl in ['상','중','하']: # very last dir
            path = f"{train_path}/{category}/{lvl}_train_{category}"
            with open(f"{path}/annotation.json","r") as anno_f:
                anno_json = json.load(anno_f)
            with open(f"{path}/images.json","r") as img_f:
                img_json = json.load(img_f)
            with open(f"{path}/question.json","r") as que_f:
                que_json = json.load(que_f)
            
            # required data
            annotations = anno_json["annotations"]
            images = img_json["images"]
            questions = que_json["questions"]
            
            # image dictionary : to be used when merging them all together, image path extraction
            _img_dict = {}
            for img_set in images:
                if img_set["image_id"] in _img_dict:
                    if img_set["image_id"] not in img_dup:
                        img_dup[img_set["image_id"]] = 1
                    elif img_set["image_id"] in img_dup:
                        img_dup[img_set["image_id"]] += 1
                _img_dict[img_set["image_id"]] = img_set["image"]
            # annotation dictionary : to be used when merging them all together, answer extraction
            _anno_dict = {}
            for anno_set in annotations:
                if anno_set["question_id"] in _anno_dict:
                    if anno_set["question_id"] not in que_id_dup:
                        que_id_dup[f'{path}-{anno_set["question_id"]}'] = 1
                    elif anno_set["question_id"] in que_id_dup:
                        que_id_dup[f'{path}-{anno_set["question_id"]}'] += 1
                _anno_dict[anno_set["question_id"]] = anno_set["multiple_choice_answer"]               
            # question dictionary : the base dictionary, image path and answer will be merged here
            for que_set in questions:
                _temp_que_dict = que_set
                _temp_que_dict["image"] = _img_dict[que_set["image_id"]]
                _temp_que_dict["answer"] = _anno_dict[que_set["question_id"]]
                train_only_item.append(_temp_que_dict)
    train_only_dict = {"annotations" : [train_only_item]}
    
    # splitting into train/valid/test
    train_item = []
    valid_item = []
    test_item = []
    
    valid_path = f"./valid_labels"
    categories = os.listdir(valid_path)
    for category in tqdm(categories):
        for lvl in ['상','중','하']: # very last dir
            path = f"{valid_path}/{category}/{lvl}_validate_{category}" ###start here again
            with open(f"{path}/annotation.json","r") as anno_f:
                anno_json = json.load(anno_f)
            with open(f"{path}/images.json","r") as img_f:
                img_json = json.load(img_f)
            with open(f"{path}/question.json","r") as que_f:
                que_json = json.load(que_f)
            
            # required data
            annotations = anno_json["annotations"]
            images = img_json["images"]
            questions = que_json["questions"]
            
            # image dictionary : to be used when merging them all together, image path extraction
            _img_dict = {}
            for img_set in images:
                if img_set["image_id"] in _img_dict:
                    if img_set["image_id"] not in img_dup:
                        img_dup[img_set["image_id"]] = 1
                    elif img_set["image_id"] in img_dup:
                        img_dup[img_set["image_id"]] += 1
                _img_dict[img_set["image_id"]] = img_set["image"]
            # annotation dictionary : to be used when merging them all together, answer extraction
            _anno_dict = {}
            for anno_set in annotations:
                if anno_set["question_id"] in _anno_dict:
                    if anno_set["question_id"] not in que_id_dup:
                        que_id_dup[f'{path}-{anno_set["question_id"]}'] = 1
                    elif anno_set["question_id"] in que_id_dup:
                        que_id_dup[f'{path}-{anno_set["question_id"]}'] += 1
                _anno_dict[anno_set["question_id"]] = anno_set["multiple_choice_answer"]               
            # question dictionary : the base dictionary, image path and answer will be merged here
            for que_set in questions:
                _temp_que_dict = que_set
                _temp_que_dict["image"] = _img_dict[que_set["image_id"]]
                _temp_que_dict["answer"] = _anno_dict[que_set["question_id"]]
                train_item.append(_temp_que_dict)
    train_dict = {"annotations" : [train_item]}
    # with open(f"{root_path}/train.json", 'w') as train_json:
    
    # splitting into train/val/test
    
    
    # putting trainset all together


if __name__ == "__main__":
    anno_maker()