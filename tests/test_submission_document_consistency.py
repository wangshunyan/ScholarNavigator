from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_materials_do_not_claim_unverifiable_soft_judgement_completion() -> None:
    template = (ROOT / "docs" / "contest" / "submission-report-template.md").read_text(
        encoding="utf-8"
    )
    checklist = (ROOT / "docs" / "contest" / "submission-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "soft Judgement 已完成内部 1000 条验证" not in template
    assert "历史 soft Judgement 运行曾被记录" in template
    assert "不能当作当前代码版本、P0/Faiss 正式成绩或赛事结果" in checklist

