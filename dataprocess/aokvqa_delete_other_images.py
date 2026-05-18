import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# =========================
# 1. 配置
# =========================
train_csv = "autodl-tmp/data_files/input_dataset/aokvqa_answer_generate_train.csv"
val_csv   = "autodl-tmp/data_files/input_dataset/aokvqa_answer_generate_val.csv"
test_csv   = "autodl-tmp/data_files/input_dataset/aokvqa_answer_generate_test.csv"

# 是否真正删除
# False -> 只预览
# True  -> 真删
DELETE_FILES = False

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 如果 CSV 里的 images 是相对路径，而你实际数据都挂在某个大根目录下，
# 就设置这个前缀；如果 CSV 里已经是完整可访问路径，就设为 "" 即可
PATH_PREFIX = "/root/"

# =========================
# 2. 工具函数
# =========================
def normalize_path(p: str) -> str:
    return os.path.normpath(str(p).strip())

def load_image_paths(csv_path: str):
    df = pd.read_csv(csv_path)
    if "images" not in df.columns:
        raise ValueError(f"{csv_path} 中不存在 images 列")
    paths = []
    for x in df["images"].dropna():
        p = normalize_path(x)
        if PATH_PREFIX:
            p = normalize_path(os.path.join(PATH_PREFIX, p))
        paths.append(p)
    return paths

def group_used_images_by_parent(image_paths):
    """
    按父目录分组：
    {
        parent_dir1: {img1.jpg, img2.jpg, ...},
        parent_dir2: {img3.jpg, img4.jpg, ...},
    }
    """
    used_map = defaultdict(set)
    for p in image_paths:
        parent = normalize_path(os.path.dirname(p))
        fname = os.path.basename(p)
        if fname:
            used_map[parent].add(fname)
    return used_map

def scan_all_images_in_dir(dir_path):
    """
    只扫描该目录下当前层的图片，不递归
    因为你说图片一般都放在最后一级文件夹
    """
    if not os.path.isdir(dir_path):
        return []

    files = []
    for fname in os.listdir(dir_path):
        full_path = os.path.join(dir_path, fname)
        if os.path.isfile(full_path):
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTS:
                files.append(normalize_path(full_path))
    return files

def clean_one_parent_dir(parent_dir, used_filenames, delete_files=False):
    all_imgs = scan_all_images_in_dir(parent_dir)

    if not all_imgs:
        print(f"[跳过] 目录不存在或没有图片: {parent_dir}")
        return {
            "dir": parent_dir,
            "all": 0,
            "keep": 0,
            "delete": 0,
        }

    to_keep = []
    to_delete = []

    for img_path in all_imgs:
        fname = os.path.basename(img_path)
        if fname in used_filenames:
            to_keep.append(img_path)
        else:
            to_delete.append(img_path)

    print(f"\n目录: {parent_dir}")
    print(f"  目录下图片总数: {len(all_imgs)}")
    print(f"  需要保留: {len(to_keep)}")
    print(f"  可删除: {len(to_delete)}")

    preview_num = min(10, len(to_delete))
    if preview_num > 0:
        print("  待删除示例:")
        for p in to_delete[:preview_num]:
            print(f"    {p}")

    if delete_files:
        for p in to_delete:
            try:
                os.remove(p)
            except Exception as e:
                print(f"    删除失败: {p} | 原因: {e}")

    return {
        "dir": parent_dir,
        "all": len(all_imgs),
        "keep": len(to_keep),
        "delete": len(to_delete),
    }

# =========================
# 3. 读取 train / val 的图片路径
# =========================
train_paths = load_image_paths(train_csv)
val_paths = load_image_paths(val_csv)
test_paths = load_image_paths(test_csv)

print(f"train.csv 中图像路径条数: {len(train_paths)}")
print(f"val.csv 中图像路径条数: {len(val_paths)}")
print(f"test.csv 中图像路径条数: {len(test_paths)}")

# =========================
# 4. 分别按父目录分组
# =========================
train_used_map = group_used_images_by_parent(train_paths)
val_used_map = group_used_images_by_parent(val_paths)
test_used_map = group_used_images_by_parent(test_paths)

print(f"\ntrain 涉及到的图片目录数: {len(train_used_map)}")
print(f"val   涉及到的图片目录数: {len(val_used_map)}")
print(f"test   涉及到的图片目录数: {len(test_used_map)}")

# =========================
# 5. 处理 train 对应目录
# =========================
print("\n================ 清理 train 对应目录 ================")
train_stats = []
for parent_dir, used_filenames in train_used_map.items():
    stat = clean_one_parent_dir(parent_dir, used_filenames, delete_files=DELETE_FILES)
    train_stats.append(stat)

# =========================
# 6. 处理 val和test 对应目录
# =========================
print("\n================ 清理 val 对应目录 ================")
val_stats = []
for parent_dir, used_filenames in val_used_map.items():
    stat = clean_one_parent_dir(parent_dir, used_filenames, delete_files=DELETE_FILES)
    val_stats.append(stat)

print("\n================ 清理 test 对应目录 ================")
test_stats = []
for parent_dir, used_filenames in test_used_map.items():
    stat = clean_one_parent_dir(parent_dir, used_filenames, delete_files=DELETE_FILES)
    test_stats.append(stat)

# =========================
# 7. 汇总
# =========================
def summarize(stats, name):
    total_all = sum(x["all"] for x in stats)
    total_keep = sum(x["keep"] for x in stats)
    total_delete = sum(x["delete"] for x in stats)
    print(f"\n{name} 汇总:")
    print(f"  总图片数: {total_all}")
    print(f"  保留数:   {total_keep}")
    print(f"  删除数:   {total_delete}")

summarize(train_stats, "train")
summarize(val_stats, "val")
summarize(test_stats, "test")

if DELETE_FILES:
    print("\n已执行删除。")
else:
    print("\n当前为预览模式，没有真正删除文件。确认无误后，把 DELETE_FILES=True 再运行。")