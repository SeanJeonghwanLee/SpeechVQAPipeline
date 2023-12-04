import os
import json
from tqdm import tqdm
from collections import defaultdict




def anno_maker(cls, data_path, tokenizer, annotation_data_path):
    img_dup = {}
    que_id_dup = {}
    final_annotations = {}
    # merging trainset only
    train_item = []
    train_path = f"./train_labels"
    categories = os.listdir(train_path)
    for category in tqdm(categories, desc="TrainSet Only"):
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
                del _temp_que_dict["image_id"]
                _temp_que_dict["image_path"] = _img_dict[que_set["image_id"]]
                _temp_que_dict["answer"] = _anno_dict[que_set["question_id"]]
                train_item.append(_temp_que_dict)
    print("For Train Set is DONE.")
    
    # splitting into train/valid/test
    valid_item = []
    test_item = []
    
    valid_path = f"./valid_labels"
    categories = os.listdir(valid_path)
    for category in tqdm(categories, desc="ValidSet Split"):
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
            q_num = len(questions) #going to divide by 3; train:valid:test==1:1:1
            
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
            q_count = 0
            for que_set in questions:
                q_count += 1
                _temp_que_dict = que_set
                del _temp_que_dict["image_id"]
                _temp_que_dict["image_path"] = _img_dict[que_set["image_id"]]
                _temp_que_dict["answer"] = _anno_dict[que_set["question_id"]]
                
                if q_count <= int(q_num/3):
                    train_item.append(_temp_que_dict)
                elif q_count <= int((2*q_num)/3):
                    valid_item.append(_temp_que_dict)
                else:
                    test_item.append(_temp_que_dict)
    #Putting train/valid/test all together
    final_annotations["train"] = {"annotations" : [train_item]}
    final_annotations["valid"] = {"annotations" : [valid_item]}
    final_annotations["test"] = {"annotations" : [test_item]}
    print("Split is DONE")
    print(f"Train Set :{len(train_item)}")
    print(f"Valid Set :{len(valid_item)}")
    print(f"Test Set :{len(test_item)}")
    print(f"Total :{len(train_item)+len(valid_item)+len(test_item)}")
    
    # Question Tokenization
    for split in tqdm(["train","valid","test"], desc="Question Tokenization"):
        items = []
        _items = final_annotations[split]["annotations"]
        
        for _item in _items:
            que_text = _item["question"]
            tokens = tokenizer.tokenize(que_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            del _item["question"]
            _item["text_segment"] = token_ids
            items.append(_item)
    
        final_annotations[split]["annotations"] = items
    print("Question Tokenization is DONE")
    
    #ans2label and label2ans
    all_major_answers = list()
    for split in tqdm(["train","valid","test"], desc="ans2label and lebel2ans"):
        _items = final_annotations[split]["annotations"]
        for _item in _items:
            all_major_answers.append(_item["answer"])

    all_major_answers = [normalize_word(word) for word in all_major_answers]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    

    
    # with open(f"{root_path}/train.json", 'w') as train_json:
    
    # splitting into train/val/test
    
    
    # putting trainset all together


if __name__ == "__main__":
    anno_maker()