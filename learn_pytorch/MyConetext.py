class MyContext:
    def __enter__(self):
        # 进入with块时执行：申请资源/修改状态
        print("进入上下文：执行初始化")
        return self  # 可选：给with ... as var传值
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出with块时执行：释放资源/恢复状态
        print("退出上下文：执行清理")
        # 返回True可吞掉异常，返回False则抛出
        return False

# 使用
with MyContext() as ctx:
    print("执行with块内的逻辑")

from contextlib import contextmanager

@contextmanager
def my_context():
    # 1. __enter__ 逻辑：进入with块时执行
    print("进入上下文：执行初始化")
    try:
        yield  # 可选：yield后的值传给with ... as var（比如yield 123）
        # 2. with块内的逻辑执行完后，回到这里
    finally:
        # 3. __exit__ 逻辑：无论是否异常，都会执行（释放资源/恢复状态）
        print("退出上下文：执行清理")

# 使用方式完全一样
with my_context():
    print("执行with块内的逻辑")