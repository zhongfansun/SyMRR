# SyMRR: Multimodal Reflective Synergy: Orchestrating Large-Small Model Collaboration for Knowledge-Based Visual Question Answering

This repository provides the implementation workflow of **SyMRR**: Multimodal Reflective Synergy: Orchestrating Large-Small Model Collaboration for Knowledge-Based Visual Question Answering


## 1. Main Workflow

The full OK-VQA workflow contains the following stages:

1. **Environment setup**
2. **Initial answer generation**
3. **Large VLM baseline answer generation**
4. **Textual reflection data preparation**
5. **Reflective CoT generation**
6. **Visual grounding description preparation**
7. **Visual reflection compression**
8. **EVF-SAM region-level visual grounding**
9. **Final SyMRR training-data construction**

The default workflow supports different large/small model combinations, including:

```text
Large VLMs:
- llava1_5
- llava1_6

Small VLMs:
- mcan_large
- mcan_small
- vinvl
- lxmert
- visualbert
```

The scripts below mainly show the reproduced settings for `vinvl`, `mcan_small`, and `mcan_large`.

## 2. Environment Setup

Enter the main code directory:

```bash
cd ms-swift-main
```

Install the required packages:

```bash
pip install -e '.[llm]'
pip install opencv-python
pip install opencv-python-headless==4.8.1.78
pip install torchscale==0.2.0
```

Create a symbolic link to the storage directory:

```bash
ln -s /root/autodl-tmp /root/ms-swift-main/
```

If you use different storage paths, please modify the dataset paths, checkpoint paths, and output paths in the corresponding shell and Python scripts.

## 3. Initial Data Processing

Generate the initial OK-VQA answers and related files:

```bash
python dataprocess/okvqa_answer_generate.py
```

Remove unused images to keep only the required image subset:

```bash
python dataprocess/delete_other_images.py
```

Move the downloaded ModelScope weights to the large VLM weight directory:

```bash
mv /root/.cache/modelscope/hub/models/swift/* /root/autodl-tmp/large_VLM_weights/
```

## 4. Generate Large VLM Baseline Answers

Run LLaVA-1.5 baseline inference:

```bash
bash okvqa_llava1_5_answer.sh
```

Run LLaVA-1.6 baseline inference:

```bash
bash okvqa_llava1_6_answer.sh
```

These outputs are used as large-only baselines and can also verify whether the large VLM inference environment is correctly configured.

## 5. Prepare Textual Reflection Data

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
lxmert
visualbert
```

Example:

```bash
python dataprocess/okvqa_text_reflective.py --model_name mcan_small
python dataprocess/okvqa_text_reflective.py --model_name mcan_large
python dataprocess/okvqa_text_reflective.py --model_name vinvl
```

## 6. Generate Textual Reflection and Visual Compression Data

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
<small_model_name>  = mcan_large, mcan_small, vinvl, lxmert, or visualbert
<split>             = train or val
<gpu_id>            = GPU index, e.g., 0
```

### 6.1 VinVL

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

### 6.2 MCAN-small

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

### 6.3 MCAN-large

#### LLaVA-1.6 + MCAN-large

```bash
bash okvqa_reflective.sh llava1_6 mcan_large train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split train
bash okvqa_compvisual.sh llava1_6 mcan_large train 0

bash okvqa_reflective.sh llava1_6 mcan_large val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split val
bash okvqa_compvisual.sh llava1_6 mcan_large val 0
```

## 7. Generate EVF-SAM Visual Grounding Results

After the visual reflection descriptions are prepared, run EVF-SAM to obtain region-level visual grounding results.

The command format is:

```bash
bash EVF-SAM-main/evf_sam.sh <gpu_id> okvqa <large_model_name> <small_model_name> <split>
```

### MCAN-large

```bash
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large val
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

## 8. Construct Final Training Data

After obtaining textual reflections, visual reflection descriptions, compressed visual data, and EVF-SAM grounding results, construct the final SyMRR training data.

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

## 9. Recommended Running Order

A typical full pipeline is:

```bash
# Step 1: environment setup
cd ms-swift-main
pip install -e '.[llm]'
pip install opencv-python
pip install opencv-python-headless==4.8.1.78
pip install torchscale==0.2.0

# Step 2: initial answer and image processing
python dataprocess/okvqa_answer_generate.py
python dataprocess/delete_other_images.py

# Step 3: large VLM baseline answers
bash okvqa_llava1_5_answer.sh
bash okvqa_llava1_6_answer.sh

# Step 4: prepare textual reflection input
python dataprocess/okvqa_text_reflective.py --model_name vinvl

# Step 5: generate textual reflection and visual compression data
bash okvqa_reflective.sh llava1_6 vinvl train 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
bash okvqa_compvisual.sh llava1_6 vinvl train 0

bash okvqa_reflective.sh llava1_6 vinvl val 0
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
bash okvqa_compvisual.sh llava1_6 vinvl val 0

# Step 6: EVF-SAM visual grounding
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl val

# Step 7: construct final training data
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
```

Replace `vinvl` with `mcan_small` or `mcan_large` to generate data for other small VLM settings.

## 10. Notes

- All scripts should be executed under `ms-swift-main`.
- The argument `<gpu_id>` is usually set to `0`, but can be changed according to your available GPU.
- The command examples use local paths such as `/root/autodl-tmp/` and `/root/ms-swift-main/`. Please update these paths according to your own environment.
- Before running `okvqa_reflective.sh`, make sure the corresponding textual reflection input file has been generated by `okvqa_text_reflective.py`.
- Before running `EVF-SAM-main/evf_sam.sh`, make sure visual reflection descriptions have been generated by `okvqa_compvisual.sh`.
- Before running `okvqa_training.py`, make sure both textual reflection and EVF-SAM visual grounding results are available.
- If you use LLaVA/LLaMA-family checkpoints, please make sure that you have obtained the required access permission and placed the checkpoints in the expected directory.


## 11. Acknowledgement

In particular, our codebase is developed based on an old version of **MS-Swift**: https://github.com/modelscope/ms-swift

We sincerely thank the authors for releasing their code and models.
