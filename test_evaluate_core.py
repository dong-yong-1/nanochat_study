import os
import sys
import torch
import random
# 将项目根目录加入Python路径（关键：解决nanochat模块导入问题）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原代码的核心依赖（从base_eval.py中复制）
from nanochat.common import print0, get_base_dir, autodetect_device_type
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.core_eval import evaluate_task  # 真实的evaluate_task实现

# 复制base_eval.py中的核心函数（保持和原代码一致）
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    import tempfile
    import shutil
    import zipfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def download_file_with_lock(url, filename, postprocess_fn=None):
    """模拟下载函数（已手动准备文件，无需真实下载）"""
    print0(f"跳过下载：{url}（已手动准备eval_bundle）")
    return

def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """完全复制base_eval.py中的evaluate_core函数"""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # 手动检查路径，方便调试
    print0(f"当前eval_bundle路径：{eval_bundle_dir}")
    if not os.path.exists(eval_bundle_dir):
        print0(f"路径不存在，尝试自动下载...")
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    # 检查关键文件是否存在
    for file_path in [config_path, eval_meta_data]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺失关键文件：{file_path}")

    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    import csv
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        if not os.path.exists(data_path):
            print0(f"\n警告：任务 {label} 的数据文件不存在：{data_path}，跳过")
            continue
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # 调用真实的evaluate_task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines.get(label, 0.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        import time
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    if centered_results:
        core_metric = sum(centered_results.values()) / len(centered_results)
    else:
        core_metric = 0.0
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -------------------------- 测试主函数 --------------------------
if __name__ == "__main__":
    # 1. 配置设备（优先CPU，避免GPU问题）
    device_type = autodetect_device_type()
    # 强制使用CPU测试（更稳定）
    device = torch.device("cpu")
    print0(f"使用设备：{device}")

    # 2. 加载测试用的模型和tokenizer（用HuggingFace的GPT2，无需训练好的nanochat模型）
    print0("加载测试模型：gpt2")
    from transformers import AutoModelForCausalLM
    # 加载轻量级模型，适合测试
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_model.to(device)
    hf_model.eval()

    # 包装为nanochat兼容的模型（复制base_eval.py中的ModelWrapper）
    class ModelWrapper:
        def __init__(self, model, max_seq_len=None):
            self.model = model
            self.max_seq_len = max_seq_len
        def __call__(self, input_ids, targets=None, loss_reduction='mean'):
            logits = self.model(input_ids).logits
            if targets is None:
                return logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        def get_device(self):
            return next(self.model.parameters()).device
    model = ModelWrapper(hf_model, max_seq_len=1024)

    # 加载tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")

    # 3. 单点测试evaluate_core函数
    try:
        print0("\n开始测试evaluate_core函数...")
        # max_per_task=10：只取前10条数据，加速测试
        results = evaluate_core(model, tokenizer, device, max_per_task=10)
        # 打印结果
        print0("\n===== 测试结果 =====")
        print0(f"各任务准确率：{results['results']}")
        print0(f"中心化结果：{results['centered_results']}")
        print0(f"CORE核心指标：{results['core_metric']:.4f}")
    except Exception as e:
        print0(f"\n测试失败：{type(e).__name__}: {e}")
        # 打印详细路径信息，方便调试
        base_dir = get_base_dir()
        eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
        print0("\n=== 路径调试信息 ===")
        print0(f"基准目录：{base_dir}")
        print0(f"eval_bundle 存在？：{os.path.exists(eval_bundle_dir)}")
        if os.path.exists(eval_bundle_dir):
            print0(f"eval_bundle 内的文件：{os.listdir(eval_bundle_dir)}")
            
            if os.path.exists(os.path.join(eval_bundle_dir, "eval_data")):
                print0(f"eval_data 内的文件：{os.listdir(os.path.join(eval_bundle_dir, 'eval_data'))}")