def should_retry(error_type):
    if error_type in ("timeout","connection"):
        return True
    return False

results = ["authentication", "ok"]
max_attempts = 2

for attempt in range(1,max_attempts + 1):
    result = results[attempt - 1]
    print(f"第{attempt}次尝试：{result}")

    if result == "ok":
        print("成功，结束。")
        break

    if not should_retry(result):
        print("此类错误不自动重试，停止。")
        break

    if attempt == max_attempts:
        print("以达到尝试次数上限，停止。")
        break

    print("允许重试，继续下一次尝试。")

def fake_request():
    raise TimeoutError("模拟请求超时")

try:
    reply = fake_request()
    print("收到回答：",reply)
except TimeoutError as exc:
    print("捕获到超时：",exc)

print("异常处理结束，程序继续。")
