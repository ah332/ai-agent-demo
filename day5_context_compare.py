"""固定真实片段的改动前后对比；不调用聊天或 Embedding 接口。"""
import json
from pathlib import Path
import sys

import config
from paper_store import MAX_CONTEXT_CHARS, PaperStore


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(__file__).resolve().parent / "验收数据"
    baseline = json.loads((folder / "day5_context_before.json").read_text(encoding="utf-8"))
    store = PaperStore()
    try:
        count_before = store.collection.count()
        expanded = store.expand_context(baseline["hits"])
        for before, after in zip(baseline["hits"], expanded):
            required = "adaptive channel enhancement" if before["page"] == 3 else "Faster R-CNN [7] (mAP 75.4%)"
            assert required in after["text"], "关键上下文尚未补全"
            assert before["paper_id"] == after["paper_id"] and before["page"] == after["page"]
            assert len(after["text"]) <= MAX_CONTEXT_CHARS
            if before["page"] == 9:
                assert after["text"].count("75.4%") == 1, "指标文字重复"
            print(f"第 {before['page']} 页：{len(before['text'])} → {len(after['text'])} 字符")
            print("原开头：", before["text"][:35])
            print("已补回：", required)
            print("实际使用片段：", after["context_chunk_ids"])
        assert store.collection.count() == count_before, "数据库片段数量改变"
        (folder / "day5_context_after.json").write_text(
            json.dumps({"question": baseline["question"], "note": "Local context comparison, no model calls.",
                        "hits": expanded}, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print("本地固定片段对比通过；未改动已有入库片段。")
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
