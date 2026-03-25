import os
import tqdm
import pyarrow.parquet as pq
from nanochat.tokenizer import get_tokenizer
from nanochat.dataset import list_parquet_files

def count_all_tokens():
    # 加载你训练好的 tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # 获取你下载的所有 parquet 文件
    parquet_files = list_parquet_files()
    print(f"找到 {len(parquet_files)} 个数据文件\n")

    total_tokens = 0
    total_docs = 0

    for filepath in parquet_files:
        print(f"正在统计: {os.path.basename(filepath)}")
        pf = pq.ParquetFile(filepath)

        for rg_idx in tqdm.tqdm(range(pf.num_row_groups)):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()

            # 对这批文本做 tokenize
            token_lists = tokenizer.encode(texts, prepend=bos_token, num_threads=4)

            # 累加 token 数
            for tokens in token_lists:
                total_tokens += len(tokens)
                total_docs += 1

    print("\n" + "="*50)
    print(f"📊 你的数据集统计结果：")
    print(f"文件数量：{len(parquet_files)}")
    print(f"总文档数：{total_docs:,}")
    print(f"总TOKENS 数：{total_tokens:,}")
    print(f"≈ {total_tokens / 1e6:.2f}M tokens")
    print("="*50)

if __name__ == "__main__":
    count_all_tokens()