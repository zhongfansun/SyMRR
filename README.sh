

scp -r -P 22 /group_file/HUyongli/szf/asimplebaseline/path_to_the_images/a_ok_vqa/test2017.zip root@connect.bjb1.seetacloud.com:/root/autodl-tmp/image_files/aokvqa/
ssh -p 35888
ssh -p 35888 root@connect.bjb1.seetacloud.com

cd ms-swift-main
pip install -e '.[llm]'
pip install opencv-python
pip install opencv-python-headless==4.8.1.78
pip install torchscale==0.2.0

ln -s /root/autodl-tmp /root/ms-swift-main/

python dataprocess/okvqa_answer_generate.py
python dataprocess/delete_other_images.py

mv /root/.cache/modelscope/hub/models/swift/* /root/autodl-tmp/large_VLM_weights/

bash okvqa_llava1_5_answer.sh
bash okvqa_llava1_6_answer.sh

# 准备反思数据
# mcan_large mcan_small vinvl lxmert visualbert
python dataprocess/okvqa_text_reflective.py --model_name vinvl


# vinvl
# 反思
bash okvqa_reflective.sh llava1_5 vinvl train 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name vinvl --output_split train
# 压缩
bash okvqa_compvisual.sh llava1_5 vinvl train 0

# 反思
bash okvqa_reflective.sh llava1_5 vinvl val 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name vinvl --output_split val
# 压缩
bash okvqa_compvisual.sh llava1_5 vinvl val 0

# 反思
bash okvqa_reflective.sh llava1_6 vinvl train 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
# 压缩
bash okvqa_compvisual.sh llava1_6 vinvl train 0

# 反思
bash okvqa_reflective.sh llava1_6 vinvl val 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name vinvl --output_split val
# 压缩
bash okvqa_compvisual.sh llava1_6 vinvl val 0


# mcan_small
# 反思
bash okvqa_reflective.sh llava1_5 mcan_small train 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_small --output_split train
# 压缩
bash okvqa_compvisual.sh llava1_5 mcan_small train 0

# 反思
bash okvqa_reflective.sh llava1_5 mcan_small val 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_5 --small_model_name mcan_small --output_split val
# 压缩
bash okvqa_compvisual.sh llava1_5 mcan_small val 0

# 反思
bash okvqa_reflective.sh llava1_6 mcan_small train 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_small --output_split train
# 压缩
bash okvqa_compvisual.sh llava1_6 mcan_small train 0

# 反思
bash okvqa_reflective.sh llava1_6 mcan_small val 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_small --output_split val
# 压缩
bash okvqa_compvisual.sh llava1_6 mcan_small val 0


# mcan_large
# 反思
bash okvqa_reflective.sh llava1_6 mcan_large train 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split train
# 压缩
bash okvqa_compvisual.sh llava1_6 mcan_large train 0

# 反思
bash okvqa_reflective.sh llava1_6 mcan_large val 0
# 准备压缩数据
python dataprocess/okvqa_compvisual.py --large_model_name llava1_6 --small_model_name mcan_large --output_split val
# 压缩
bash okvqa_compvisual.sh llava1_6 mcan_large val 0



bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_large val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_small train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 mcan_small val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_small train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 mcan_small val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_5 vinvl val

bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl train
bash EVF-SAM-main/evf_sam.sh 0 okvqa llava1_6 vinvl val


python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_large --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_large --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_small --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name mcan_small --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_small --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name mcan_small --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_6 --small_model_name vinvl --output_split val

python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name vinvl --output_split train
python dataprocess/okvqa_training.py --large_model_name llava1_5 --small_model_name vinvl --output_split val

