import json
import os
import pandas as pd
import argparse


def main():
    for output_split in ['val']:

        with open('autodl-tmp/data_files/okvqa_problems.json', 'r', encoding='utf-8') as f:
            problems = json.load(f)

        with open(
            "autodl-tmp/output_files/okvqa_llava1_5_answer_val/20260421-103424.jsonl",
            'r',
            encoding='utf-8'
        ) as f:
            answer_candidates_json = json.load(f)

        answer_candidates = {
            str(item["question_id"]): item["answer"]
            for item in answer_candidates_json
        }

        query = []
        response = []
        images = []

        for _, single_sample in problems.items():
            if single_sample['split'] == output_split:
                qid = str(single_sample['question_id'])
                preds = answer_candidates[qid]
                lmm_answer = str(preds).lower()

                query_item = (
                    '<image>' + single_sample['question']
                    + f' Answer: {lmm_answer}.'
                    + ' Please explain the answer briefly with one short paragraph.'
                )
                query.append(query_item)

                image_str = (
                    f'COCO_{output_split}2014_'
                    + str(single_sample['image_id']).zfill(12)
                    + '.jpg'
                )
                images_item = os.path.join(
                    f'autodl-tmp/image_files/okvqa/{output_split}2014',
                    image_str
                )
                images.append(images_item)

                response_item = []
                response.append(response_item)

        llama_preds_df = pd.DataFrame({
            'query': query,
            'response': response,
            'images': images
        })

        save_path = f'autodl-tmp/data_files/input_dataset/okvqa_llava1_5_reflective_{output_split}.csv'
        llama_preds_df.to_csv(save_path, index=False)
        print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()