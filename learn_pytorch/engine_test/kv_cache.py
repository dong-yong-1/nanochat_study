import torch
class SimpleKVCache:
    def __init__(self, n_layers, batch_size, max_seq_len, n_heads, head_dim, device):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_head = n_heads
        self.head_dim = head_dim
        self.device = device

        self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim, device=device)
        self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim, device=device)

        self.current_length = torch.zeros(batch_size, device=device, dtype=torch.long)

    def update(self, layer_id, new_k, new_v):
        """
        更新指定层的缓存，将新的键值对添加到当前位置
        new_k, new_v: [batch_size, n_head, head_dim]
        """
        pos = self.current_length[0].item()
        if pos >= self.max_seq_len:
            raise RuntimeError("KV Cache 已满！")

        self.k_cache[layer_id,:,pos:pos + 1,:,:] = new_k
        self.v_cache[layer_id,:,pos:pos + 1,:,:] = new_v

        self.current_length += 1

    def get_full_cache(self, layer_id):
        """
        获取指定层的完整缓存
        返回: [batch_size, seq_len, n_head, head_dim]
        """
        pos = self.current_length[0].item()
        return self.k_cache[layer_id,:,:pos,:,:], self.v_cache[layer_id,:,:pos,:,:]

    def reset(self):
        """
        重置缓存，清空所有内容
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_length.zero_()

if __name__ == "__main__":
    cache = SimpleKVCache(n_layers=2, max_seq_len=10, n_heads=4, head_dim=16, batch_size=2, device='cpu')
    
    # 模拟生成过程
    for step in range(5):
        # 假设模型刚算出了新的 K, V (形状：Batch=2, Len=1, Heads=4, Dim=16)
        fake_k = torch.randn(2, 1, 4, 16)
        fake_v = torch.randn(2, 1, 4, 16)
        
        # 更新到缓存 (每一层都要更新，这里只演示第 0 层)
        cache.update(0, fake_k, fake_v)
        
        print(f"Step {step}: 当前长度 = {cache.current_length[0].item()}")
        
        # 验证：取出来的形状应该是 (2, step+1, 4, 16)
        k_full, v_full = cache.get_full_cache(0)
        assert k_full.shape[1] == step + 1, "长度不对!"
        
    print("KV Cache 测试通过！✅")

        
