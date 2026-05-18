import json
import os
import pandas as pd

#生成lmm的答案的输入数据
for output_split in ['train', 'val', 'test']:
    problems = json.load(open('autodl-tmp/data_files/aokvqa_problems.json'))

    system = []
    query = []
    response = []
    images = []

    for split, single_sample in problems.items():
        if single_sample['split'] == output_split:
            # system_item = 'Answer the question using a single word or phrase.'
            # system.append(system_item)

            query_item = '<image>' + single_sample['question'] + ' Answer the question using a single word or phrase.'
            query.append(query_item)

            image_str = '0' * (12 - len(list(str(single_sample['image_id'])))) + str(single_sample['image_id']) + '.jpg'
            images_item = os.path.join(f'autodl-tmp/image_files/aokvqa/{output_split}2017', image_str)
            images.append(images_item)

            response_item = []
            response.append(response_item)

    llama_preds_df = pd.DataFrame(
        {'query': query, 'response': response, 'images': images}) #'system': system,
    llama_preds_df.to_csv(f'autodl-tmp/data_files/input_dataset/aokvqa_answer_generate_{output_split}.csv', index=False)



