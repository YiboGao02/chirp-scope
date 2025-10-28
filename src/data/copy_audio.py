import os
import shutil

src_root = r'dataset\train_data\input_data'  # 你的原始数据根目录
dst_root = r'dataset\train_data\train_audios'  # 目标目录

os.makedirs(dst_root, exist_ok=True)

for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith('.ogg'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_root, file)
            shutil.copy2(src_path, dst_path)  # 或用shutil.move(src_path, dst_path)来移动