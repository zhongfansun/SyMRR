import json
import os
import pandas as pd

# 生成lmm的答案的输入数据
for output_split in ['val']:
    problems = json.load(open('autodl-tmp/data_files/okvqa_problems.json'))

    FILE_PATH_new = "autodl-tmp/output_files/okvqa_lmmcot_val.jsonl"

    preds = []
    with open(FILE_PATH_new, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            preds.append(data['response'])

    system = []
    query = []
    response = []
    images = []
    i = 0
    for split, single_sample in problems.items():
        if single_sample['split'] == output_split:
            # system_item = 'Answer the question using a single word or phrase.'
            # system.append(system_item)
            pred = preds[i]
            i += 1
            query_item = (
                    "<image>"
                    + f"Explanation: {pred}. "
                    + single_sample["question"]
                    + ' Answer the question using a single word or phrase.'
            )

            query.append(query_item)

            image_str = f'COCO_{output_split}2014_' + '0' * (12 - len(str(single_sample['image_id']))) + str(
                single_sample['image_id']) + '.jpg'
            images_item = os.path.join(f'autodl-tmp/image_files/okvqa/{output_split}2014', image_str)
            images.append(images_item)

            response_item = []
            response.append(response_item)

    llama_preds_df = pd.DataFrame(
        {'query': query, 'response': response, 'images': images})  # 'system': system,
    llama_preds_df.to_csv(f'autodl-tmp/data_files/input_dataset/okvqa_answer_self_reflect_generate_{output_split}.csv', index=False)



