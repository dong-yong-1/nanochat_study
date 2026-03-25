import os
import sys
import yaml
import csv
import random
import time
import json
# 把项目根目录加入Python路径，确保能导入内部函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nanochat.common import download_file_with_lock, get_base_dir
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(zip_path):
    """解压eval_bundle.zip（代码里的postprocess_fn，需和原代码一致）"""
    import zipfile
    base_dir = get_base_dir()
    print(f"base_dir: {base_dir}")
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    os.makedirs(eval_bundle_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(eval_bundle_dir)
    # 解压后删除zip文件（可选）
    os.remove(zip_path)

def test_download_eval_bundle():
    """单独测试下载eval_bundle数据集"""
    print("开始下载CORE评估数据集...")
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    
    # 1. 检查是否已下载，避免重复下载
    if os.path.exists(eval_bundle_dir):
        print(f"✅ 数据集已存在：{eval_bundle_dir}")
        # 验证文件完整性
        required_files = [
            os.path.join(eval_bundle_dir, "core.yaml"),
            os.path.join(eval_bundle_dir, "eval_meta_data.csv"),
            os.path.join(eval_bundle_dir, "eval_data")
        ]
        all_exist = all(os.path.exists(f) for f in required_files)
        if all_exist:
            print("✅ 数据集文件完整，无需重新下载")
            return
        else:
            print("⚠️ 数据集文件不完整，重新下载...")
    
    # 2. 执行下载（和原代码逻辑一致）
    try:
        download_file_with_lock(
            url=EVAL_BUNDLE_URL,
            filename="eval_bundle.zip",
            postprocess_fn=place_eval_bundle  # 解压函数
        )
        print("✅ 数据集下载并解压完成！")
        
        # 3. 验证下载结果
        if os.path.exists(eval_bundle_dir):
            print(f"📂 数据集路径：{eval_bundle_dir}")
            # 打印目录结构，确认文件齐全
            for root, dirs, files in os.walk(eval_bundle_dir):
                level = root.replace(eval_bundle_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                sub_indent = ' ' * 2 * (level + 1)
                for f in files[:5]:  # 只打印前5个文件，避免刷屏
                    print(f'{sub_indent}{f}')
    except Exception as e:
        print(f"❌ 下载失败：{type(e).__name__}: {e}")
        # 清理不完整的文件
        if os.path.exists("eval_bundle.zip"):
            os.remove("eval_bundle.zip")
        if os.path.exists(eval_bundle_dir):
            import shutil
            shutil.rmtree(eval_bundle_dir)
        sys.exit(1)

if __name__ == "__main__":
    test_download_eval_bundle()