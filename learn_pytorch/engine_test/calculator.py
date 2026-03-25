import signal
import warnings
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    def handler(sig, frame):
        raise TimeoutError("Operation timed out")
        
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def safe_eval(expr):
    expr = expr.replace(",","")

    allowed = set("0123456789+-*/().%  \"'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    if not all(c in allowed for c in expr):
        return None

    dangerous = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'os', 'sys']
    if any(d in expr for d in dangerous):
        return None

    is_math = all(c in "0123456789+-*/().% " for c in expr)
    is_count = ".count(" in expr and is_math == False

    if not (is_math or is_count):
        return None

    try:
        with timeout(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return eval(expr, {"__builtins__": {}},{})
    except Exception:
        return None

# --- 测试 ---
if __name__ == "__main__":
    tests = [
        "2 + 2", 
        "100 * 5", 
        "'hello world'.count('o')", 
        "import os", # 应被拦截
        "while True: pass", # 应超时
        "2 ** 100" # 原代码禁止了幂运算，这里你可以选择是否禁止
    ]
    
    for t in tests:
        res = safe_eval(t)
        print(f"Input: {t:<30} | Result: {res}")    