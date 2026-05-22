"""Keyword-based hidden contribution recognizer for MVP demo use."""

from __future__ import annotations

import re


KEYWORD_MAP = {
    "跨部门协作": ["跨部门", "协同", "协调", "联动", "产品", "客服", "业务", "财务", "法务", "运营"],
    "技术攻关": ["技术攻关", "故障", "性能", "稳定性", "架构", "优化", "排查", "难题", "算法", "上线"],
    "带教新人": ["带教", "新人", "导师", "培养", "交接", "辅导", "陪跑", "培训"],
    "知识分享": ["知识分享", "分享会", "文档", "复盘", "沉淀", "课程", "手册", "最佳实践"],
    "冲突处理": ["冲突", "分歧", "投诉", "协调矛盾", "争议", "沟通", "谈判"],
    "组织影响力": ["影响力", "推动", "牵头", "组织", "机制", "标准", "共识", "委员会"],
    "危机处理": ["危机", "紧急", "事故", "应急", "高频故障", "舆情", "风险", "止损"],
    "流程优化": ["流程", "效率", "自动化", "标准化", "降本", "提效", "缩短", "减少", "优化"],
}


OUTCOME_KEYWORDS = ["提升", "降低", "减少", "缩短", "解决", "完成", "上线", "落地", "恢复", "达成"]


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def detect_hidden_contributions(text: str) -> dict:
    cleaned_text = re.sub(r"\s+", "", text or "")

    if not cleaned_text:
        return {
            "tags": [],
            "summary": "尚未输入贡献描述，暂无法识别隐性贡献。",
            "score": 0,
            "needs_review": False,
            "validation_note": "HR 需要结合项目记录、同事反馈和管理者访谈验证材料真实性。",
        }

    tag_hits: dict[str, int] = {}
    for tag, keywords in KEYWORD_MAP.items():
        hits = _count_keyword_hits(cleaned_text, keywords)
        if hits:
            tag_hits[tag] = hits

    tags = list(tag_hits.keys())
    evidence_bonus = min(15, _count_keyword_hits(cleaned_text, OUTCOME_KEYWORDS) * 3)
    length_bonus = min(15, len(cleaned_text) // 35 * 3)
    diversity_bonus = min(35, len(tags) * 7)
    depth_bonus = min(20, sum(tag_hits.values()) * 2)
    score = min(100, 20 + diversity_bonus + depth_bonus + evidence_bonus + length_bonus)
    if not tags:
        score = min(45, 20 + length_bonus)

    if tags:
        tag_text = "、".join(tags)
        summary = f"描述中体现出{tag_text}等隐性贡献，说明该员工可能在正式 KPI 之外承担了协作、推动或组织支持类工作。"
    else:
        summary = "描述中暂未识别到明确的隐性贡献标签，建议补充项目背景、个人角色、协作对象和结果证据。"

    needs_review = score >= 80 or ("危机处理" in tags and "跨部门协作" in tags)

    return {
        "tags": tags,
        "summary": summary,
        "score": int(score),
        "needs_review": needs_review,
        "validation_note": "该识别结果仅用于辅助发现线索，HR 需要人工验证项目记录、协作方反馈和材料真实性。",
    }
