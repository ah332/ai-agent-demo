"""第 5 天：读取真实相邻片段，用用户编写的函数合并；不调用模型。"""
import sys

import config  # 加载数据库相关配置，不输出或使用密钥。
from paper_store import PaperStore
from paper_text import merge_overlapping


PAPER_ID = "5437dd846434733ccc10da1f9522c5ace39f828907d349898bc2e083f42fad98"


def main():
    # 避免终端编码不支持 PDF 中的特殊字符，影响本地阅读。
    sys.stdout.reconfigure(encoding="utf-8")
    store = PaperStore()
    try:
        previous_id = f"{PAPER_ID}:p9:c3"
        current_id = f"{PAPER_ID}:p9:c4"
        records = store.collection.get(
            ids=[previous_id, current_id],
            include=["documents", "metadatas"],
        )
        # 按编号取文字，不假定数据库返回顺序与请求顺序相同。
        texts = dict(zip(records["ids"], records["documents"]))
        if previous_id not in texts or current_id not in texts:
            raise ValueError("本机未找到目标片段，请先确认原 YOLO-EMAC.pdf 已入库。")
        for metadata in records["metadatas"]:
            if metadata["paper_id"] != PAPER_ID or metadata["page"] != 9:
                raise ValueError("片段来源不符合本次练习要求。")

        left = texts[previous_id]
        right = texts[current_id]
        merged = merge_overlapping(left, right)

        print("读取对象：YOLO-EMAC.pdf，第 9 页 c3 与 c4")
        print("原 c4 开头：", right[:80])
        print("左片段长度：", len(left))
        print("右片段长度：", len(right))
        print("合并长度：", len(merged))
        print("去掉的重叠字符数：", len(left) + len(right) - len(merged))
        print("75.4% 出现次数：", merged.count("75.4%"))
        print("合并后的原文：")
        print(merged)
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
