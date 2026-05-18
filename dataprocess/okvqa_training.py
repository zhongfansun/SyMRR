import json
import os
import argparse
import pandas as pd

def get_latest_file(dir_path):
    """返回目录中最新修改的文件路径"""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")

    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

    if not files:
        raise FileNotFoundError(f"目录中没有文件: {dir_path}")

    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large_model_name", type=str, required=True)
    parser.add_argument("--small_model_name", type=str, required=True)
    parser.add_argument("--output_split", type=str, required=True, choices=["train", "val"])
    args = parser.parse_args()

    large_model_name = args.large_model_name
    small_model_name = args.small_model_name
    output_split = args.output_split

    # 1. 构造输入目录
    FILE_PATH = (
        f"autodl-tmp/output_files/"
        f"okvqa_{large_model_name}_{small_model_name}_reflective_{output_split}/"
    )

    # 2. 读取该目录下最新文件
    FILE_PATH_new = get_latest_file(FILE_PATH)
    print(f"读取最新文件: {FILE_PATH_new}")

    # 3. 读取 problems
    problems_path = "autodl-tmp/data_files/okvqa_problems.json"
    with open(problems_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # 4. 读取最新文件中的 response
    preds = []
    with open(FILE_PATH_new, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            preds.append(data['response'])

    query = []
    response = []
    images = []

    i = 0
    max_words = 120
    for _, single_sample in problems.items():
        if single_sample['split'] == output_split:
            explain = preds[i]

            words = explain.split()
            if len(words) > max_words:
                explain = " ".join(words[:max_words])

            i = i + 1
            # Explanation: {explain}\n
            query_item = '<image>' + \
                         f' Explanation: {explain}\n' + \
                         single_sample['question'] + ' Answer the question using a single word or phrase.'
            query.append(query_item)

            image_str = (
                    f"COCO_{output_split}2014_"
                    + str(single_sample["image_id"]).zfill(12)
                    + ".jpg"
            )
            images_item = os.path.join(
                f"autodl-tmp/image_files/okvqa/{output_split}2014",
                image_str
            )

            image_str_roi = str(single_sample['question_id'])+'_'+image_str
            images_item_roi = os.path.join(
                f"autodl-tmp/seg_image_files/okvqa/okvqa_{large_model_name}_{small_model_name}_{output_split}",
                image_str_roi
            )

            each_images = []
            each_images.append(images_item)
            each_images.append(images_item_roi)
            images.append(each_images)

            GT_ans_item = single_sample['direct_answers']
            response_item = max(set(GT_ans_item), key=GT_ans_item.count)
            response.append(response_item)

    # 6. 保存 csv
    save_path = (
        f"autodl-tmp/data_files/input_dataset/"
        f"okvqa_{large_model_name}_{small_model_name}_answer_{output_split}.csv"
    )

    llama_preds_df = pd.DataFrame({
        "query": query,
        "response": response,
        "images": images
    })
    llama_preds_df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"保存完成: {save_path}")
    print(f"共写入 {len(llama_preds_df)} 条数据")


if __name__ == "__main__":
    main()
