import argparse
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide



def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--vis_save_path", default="/HUyongli/szf/data/aaai2026/seg_images_train_llava1_5_13b", type=str)
    parser.add_argument("--prompt", type=str, default="/HUyongli/szf/data/aaai2026/mcan_compvisual_val_llava1_5_13b/20250718-023244.jsonl")
    parser.add_argument("--image_path", type=str, default="/group_file/HUyongli/szf/asimplebaseline/path_to_the_images/ok_vqa/train2014/")
    parser.add_argument("--problems_file", type=str, default="autodl-tmp/data_files/okvqa_problems.json")
    parser.add_argument("--splits_file", type=str, default="autodl-tmp/data_files/okvqa_pid_splits.json")
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    parser.add_argument("--version", type=str, default="autodl-tmp/other_weights/evf-sam-multitask")
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def main(args):
    args = parse_args(args)

    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.vis_save_path, exist_ok=True)
    problems = json.load(open(args.problems_file))
    splits = json.load(open(args.splits_file))[args.splits]
    questions = [problems[qid] for qid in splits]

    prompt_list = []
    with open(args.prompt, 'r',
              encoding='utf-8') as f:
        # 逐行读取并解析
        for line in f:
            data = json.loads(line)
            prompt_list.append(data['response'])

    image_path_list = []
    save_path_list = []
    for each_question in questions:
        image_context = f'COCO_{args.splits}2014_' + '0' * (12 - len(str(each_question['image_id']))) + str(each_question['image_id']) + '.jpg'
        seg_image_context = str(each_question['question_id']) + '_' + image_context
        image_path_list.append(args.image_path + image_context)
        save_path_list.append(os.path.join(args.vis_save_path, seg_image_context))

    # initialize model and tokenizer
    tokenizer, model = init_models(args)

    for prompt, image_path, save_path in tqdm(zip(prompt_list, image_path_list, save_path_list),
                                              total=len(prompt_list)):
        # preprocess
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)

        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

        # infer
        pred_mask = model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask < 0

        # save visualization
        save_img = image_np.copy()
        # save_img[pred_mask] = (
        #     image_np * 0.5
        #     + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        # )[pred_mask]
        save_img[pred_mask] = [255, 255, 255]  # 直接把 mask 区域设为黑色

        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, save_img)

def occupy_gpu_memory(gpu_id=0, reserve_ratio=0.45):
    device = torch.device(f"cuda:{gpu_id}")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    block_mem = int(total_memory * reserve_ratio)
    num_float32 = block_mem // 4  # 每个 float32 占 4 bytes
    x = torch.empty(num_float32, dtype=torch.float32, device=device)
    del x

if __name__ == "__main__":
    # occupy_gpu_memory()
    main(sys.argv[1:])