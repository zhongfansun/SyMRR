# SyMRR: Multimodal Reflective Synergy: Orchestrating Large-Small Model Collaboration for Knowledge-Based Visual Question Answering

This repository provides the implementation workflow of **SyMRR**: Multimodal Reflective Synergy: Orchestrating Large-Small Model Collaboration for Knowledge-Based Visual Question Answering


## 1. Main Workflow

The full OK-VQA workflow contains the following stages:

1. **Environment setup**
2. **Textual reflection data preparation**
3. **Reflective CoT generation**
4. **Visual grounding description preparation**
5. **Visual reflection compression**
6. **EVF-SAM region-level visual grounding**
7. **Final SyMRR training-data construction**

The default workflow supports different large/small model combinations, including:

```text
Large VLMs:
- llava1_5
- llava1_6

Small VLMs:
- mcan_large
- mcan_small
- vinvl
```

The scripts below mainly show the reproduced settings for `vinvl`, `mcan_small`, and `mcan_large`.

## 2. Environment Setup

Enter the main code directory:

```bash
cd SyMRR
```

Install the required packages:

```bash
pip install -e '.[llm]'
pip install opencv-python
pip install opencv-python-headless==4.8.1.78
pip install torchscale==0.2.0
```

## 3. Prepare Textual Reflection Data

Before generating reflective reasoning, prepare the reflection input data for the selected small model.

For example, for `vinvl`:

```bash
python dataprocess/okvqa_text_reflective.py --model_name vinvl
```

Supported small model names include:

```text
mcan_large
mcan_small
vinvl
```

Example:

```bash
python dataprocess/okvqa_text_reflective.py --model_name mcan_small
python dataprocess/okvqa_text_reflective.py --model_name mcan_large
python dataprocess/okvqa_text_reflective.py --model_name vinvl
```

## 4. Generate Textual Reflection and Visual Compression Data

For each large/small model combination and each data split, the workflow is:

```bash
# 1. Generate textual reflection
bash okvqa_reflective.sh <large_model_name> <small_model_name> <split> <gpu_id>

# 2. Prepare visual compression data
python dataprocess/okvqa_compvisual.py \
    --large_model_name <large_model_name> \
    --small_model_name <small_model_name> \
    --output_split <split>

# 3. Generate compressed visual reflection data
bash okvqa_compvisual.sh <large_model_name> <small_model_name> <split> <gpu_id>
```

where:

```text
<large_model_name>  = llava1_5 or llava1_6
<small_model_name>  = mcan_large, mcan_small, or vinvl
<split>             = train or val
<gpu_id>            = GPU index, e.g., 0
```

### 4.1 VinVL

#### LLaVA-1.5 + VinVL

```bash
bash okvqa_reflective.sh llava1_5 vinvl train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name vinvl --output_split train
bash okvqa_compvisual.sh llava1_5 vinvl train 0

bash okvqa_reflective.sh llava1_5 vinvl val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name vinvl --output_split val
bash okvqa_compvisual.sh llava1_5 vinvl val 0
```

#### LLaVA-1.6 + VinVL

```bash
bash okvqa_reflective.sh llava1_6 vinvl train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
bash okvqa_compvisual.sh llava1_6 vinvl train 0

bash okvqa_reflective.sh llava1_6 vinvl val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
bash okvqa_compvisual.sh llava1_6 vinvl val 0
```

### 4.2 MCAN-small

#### LLaVA-1.5 + MCAN-small

```bash
bash okvqa_reflective.sh llava1_5 mcan_small train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_small --output_split train
bash okvqa_compvisual.sh llava1_5 mcan_small train 0

bash okvqa_reflective.sh llava1_5 mcan_small val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_small --output_split val
bash okvqa_compvisual.sh llava1_5 mcan_small val 0
```

#### LLaVA-1.6 + MCAN-small

```bash
bash okvqa_reflective.sh llava1_6 mcan_small train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_small --output_split train
bash okvqa_compvisual.sh llava1_6 mcan_small train 0

bash okvqa_reflective.sh llava1_6 mcan_small val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_small --output_split val
bash okvqa_compvisual.sh llava1_6 mcan_small val 0
```

### 4.3 MCAN-large

#### LLaVA-1.5 + MCAN-large

```bash
bash okvqa_reflective.sh llava1_5 mcan_large train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_large --output_split train
bash okvqa_compvisual.sh llava1_5 mcan_large train 0

bash okvqa_reflective.sh llava1_5 mcan_large val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_large --output_split val
bash okvqa_compvisual.sh llava1_5 mcan_large val 0
```

#### LLaVA-1.6 + MCAN-large

```bash
bash okvqa_reflective.sh llava1_6 mcan_large train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split train
bash okvqa_compvisual.sh llava1_6 mcan_large train 0

bash okvqa_reflective.sh llava1_6 mcan_large val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split val
bash okvqa_compvisual.sh llava1_6 mcan_large val 0
```

## 5. Generate EVF-SAM Visual Grounding Results

After the visual reflection descriptions are prepared, run EVF-SAM to obtain region-level visual grounding results.

The command format is:

```bash
bash EVF-SAM-main/evf_sam.sh <gpu_id> okvqa <large_model_name> <small_model_name> <split>
```

### MCAN-large

```bash
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_large train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_large val
```

### MCAN-small

```bash
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_small train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_small val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_small train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_small val
```

### VinVL

```bash
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 vinvl val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl val
```

## 6. Construct Final Training Data

After obtaining textual reflections, and EVF-SAM grounding, construct the final SyMRR training data.

The command format is:

```bash
python dataprocess/okvqa_training.py \
    --large_model_name <large_model_name> \
    --small_model_name <small_model_name> \
    --output_split <split>
```

### MCAN-large

```bash
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_large --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_large --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_large --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_large --output_split val
```

### MCAN-small

```bash
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_small --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_small --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_small --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_small --output_split val
```

### VinVL

```bash
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name vinvl --output_split val
```

## 7. Recommended Running Order

A typical full pipeline is:

```bash
# Step 1: environment setup
cd ms-swift-main
pip install -e '.[llm]'
pip install opencv-python
pip install opencv-python-headless==4.8.1.78
pip install torchscale==0.2.0

# Step 2: prepare textual reflection input
python dataprocess/okvqa_text_reflective.py --model_name vinvl

# Step 3: generate textual reflection and visual compression data
bash okvqa_reflective.sh llava1_6 vinvl train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
bash okvqa_compvisual.sh llava1_6 vinvl train 0

bash okvqa_reflective.sh llava1_6 vinvl val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
bash okvqa_compvisual.sh llava1_6 vinvl val 0

# Step 4: EVF-SAM visual grounding
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl val

# Step 5: construct final training data
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
```

Replace `vinvl` with `mcan_small` or `mcan_large` to generate data for other small VLM settings.


To make reproduction easier, we also provide the processed data and related files through Baidu Netdisk.

Baidu Netdisk link:

```text
https://pan.baidu.com/s/12Efaihm3SNc3RZRWupkbLw?pwd=wqw5
```

Extraction code:

```text
wqw5
```


## 8. Acknowledgement

Our code is built upon an old version of Swift:

[https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)

We sincerely thank the authors and contributors of Swift for their excellent open-source work.

We also thank the authors of Prophet for providing useful resources for MCAN-based VQA feature extraction:

[https://github.com/MILVLG/prophet](https://github.com/MILVLG/prophet)


---

## Contact

If you have any questions about the code or data preparation, please open an issue in this repository or contact the authors.
