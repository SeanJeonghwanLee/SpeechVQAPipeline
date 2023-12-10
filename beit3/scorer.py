import json


#labels 2 answers
label2ans = {}
with open("/home/seanlee/class/SpeechVQAPipeline/answer2label.txt", "r", encoding="utf-8") as ans2label:
    for i, line in enumerate(ans2label):
        data = json.loads(line)
        ans = data["answer"]
        label = data["label"]
        label = int(label)
        label2ans[label] = ans

# answer sheet
answer_dict = {}
answer_data = []
with open("/home/seanlee/class/SpeechVQAPipeline/vqa.test_ans.json", "r") as answer_sheet:

    for line in answer_sheet:
        answer_data.append(json.loads(line))

for a in answer_data:
    question_id = a["qid"]
    answer = a['labels']
    answer_dict[question_id] = answer


# predicted
with open("/home/seanlee/class/SpeechVQAPipeline/beit3/results/submit_vqacustom_test_epoch4.json","r", encoding="utf-8") as prediction_sheet:
    predictions = json.load(prediction_sheet)

correct = 0
missing = 0
total_q = len(predictions)
for pred in predictions:
    qid = str(pred["question_id"])
    ans_hat = pred["answer"]
    try:
        ans = answer_dict[qid]
    except:
        missing += 1
        continue
    
    real_ans = label2ans[ans[0]]
    
    if ans_hat == real_ans:
        print(ans_hat)
        correct += 1

statistics = {
    "total questions" : total_q,
    "correct" : correct,
    "wrong" : total_q - correct - missing,
    "missing" : missing,
    "accuracy" : correct / (total_q - missing)
}

print(statistics)
    