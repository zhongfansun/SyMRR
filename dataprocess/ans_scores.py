import json
import os
import pandas as pd
import sys

# 获取命令行参数
if len(sys.argv) < 2:
    print("Usage: python ans_score.py <path_to_json>")
    sys.exit(1)

file_path = sys.argv[1]  # 获取第一个参数作为文件路径

def direct_scores(pred_answer, direct_answers):
    acc_num = 0
    cnt = 0
    for _, answer_id in enumerate(direct_answers):
        if pred_answer == answer_id:
            cnt += 1
    if cnt ==1:
        acc_num = 0.3
    elif cnt == 2:
        acc_num = 0.6
    elif cnt > 2:
        acc_num = 1
    return acc_num

problems = json.load(open('autodl-tmp/data_files/okvqa_problems.json'))
preds = []
with open(file_path, 'r', encoding='utf-8') as f:
    # 逐行读取并解析
    for line in f:
        data = json.loads(line)
        preds.append(data['response'].strip().lower())
# preds = json.load(open('okvqa_ours_QGA_lmmcot_wans_ori.json'))['preds']
###
i = 0
score = []
for split, single_sample in problems.items():
    if single_sample['split'] == 'val':
        GT_ans_item = single_sample['direct_answers']
        score.append(direct_scores(preds[i].lower(), GT_ans_item))
        i = i+1
assert len(score)==5046
print(sum(score)/len(score))





