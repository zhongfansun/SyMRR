import os
from PIL import Image, ImageChops

# 输入路径和输出路径
input_dir = "/HUyongli/szf/data/aaai2026/images/sentence_train_sam_multitask/"
output_dir = "/HUyongli/szf/data/aaai2026/images/white_sentence_train_sam_multitask/"
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

def crop_white_border(img: Image.Image) -> Image.Image:
    """裁剪图片白边"""
    img = img.convert("RGB")  # 确保为RGB格式
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()  # 获取非白色区域的边界框
    return img.crop(bbox) if bbox else img

def process_images(input_dir: str, output_dir: str):
    """遍历输入文件夹，裁剪白边后保存到输出文件夹"""
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            cropped_img = crop_white_border(img)
            save_path = os.path.join(output_dir, filename)
            cropped_img.save(save_path)
            print(f"已处理: {filename}")

if __name__ == "__main__":
    process_images(input_dir, output_dir)
    print(f"所有图片已处理完成，保存在: {output_dir}")
