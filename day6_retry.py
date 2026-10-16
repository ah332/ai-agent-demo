def fake_request(result):
    if result == "timeout":
        raise TimeoutError("模拟请求超时")
    if result == "authentication":
        raise PermissionError("模拟认证失败")
    return "模拟回答"


results = ["authentication", "ok"]
max_attempts = 2

for attempt in range(1, max_attempts + 1):
    print(f"第 {attempt} 次尝试")

    try:
        reply = fake_request(results[attempt - 1])
        print("收到回答：", reply)
        break
    except TimeoutError as exc:
        print("捕获到超时：", exc)

        if attempt == max_attempts:
            print("已达到尝试次数上限，停止。")
            break

        print("允许重试，继续下一次尝试。")

    except PermissionError as exc:
        print("捕获到认证错误：", exc)
        print("请检查配置，停止自动重试。")
        break
