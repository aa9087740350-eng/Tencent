"""Scoring utilities for TalentLens AI.

The model is intentionally transparent and rule-based. It is designed for a
demo where HR can inspect why an employee is recommended, routed to review, or
asked to build a development plan.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


ROLE_MODELS = {
    "技术负责人": [
        ("KPI 完成情况", "kpi_score", 0.25),
        ("技术攻关", "technical_problem_solving", 0.30),
        ("项目难度", "project_difficulty", 0.20),
        ("跨部门协作", "cross_functional_collaboration", 0.15),
        ("知识分享 / 带教新人", ("knowledge_sharing", "mentoring"), 0.10),
    ],
    "团队管理者": [
        ("团队绩效", "kpi_score", 0.25),
        ("团队满意度", "team_satisfaction", 0.25),
        ("人才培养", "mentoring", 0.20),
        ("跨部门协作", "cross_functional_collaboration", 0.15),
        ("冲突处理 / 组织判断", ("conflict_resolution", "organizational_judgment"), 0.15),
    ],
    "HRBP": [
        ("员工满意度", "team_satisfaction", 0.25),
        ("业务支持效果", "business_support", 0.25),
        ("员工关系处理", "employee_relations", 0.20),
        ("项目推动能力", ("cross_functional_collaboration", "organizational_judgment"), 0.15),
        ("数据分析能力", "data_analysis", 0.15),
    ],
}


ACTION_MAP = {
    "AI 总评分与经理评分差异超过 20 分": "要求直属经理进行校准",
    "隐性贡献分高于 85，但总评分低于 75": "复查项目贡献证据",
    "KPI 分数高于 85，但岗位匹配分低于 65": "收集 360 度反馈",
    "申请团队管理者岗位，但管理能力证据分低于 60": "安排 HRBP 访谈",
    "高潜人才暂未被推荐晋升": "进入晋升委员会讨论",
    "总评分位于 65-79 分灰区": "制定发展计划",
    "风险等级不是低风险": "安排 HRBP 访谈",
}


SCORE_COLUMNS = [
    "explicit_performance_score",
    "role_match_score",
    "hidden_contribution_score",
    "development_potential_score",
    "total_score",
]


def _avg(row: pd.Series, fields: Iterable[str]) -> float:
    values = [float(row[field]) for field in fields]
    return sum(values) / len(values)


def _field_score(row: pd.Series, field: str | tuple[str, ...]) -> float:
    if isinstance(field, tuple):
        return _avg(row, field)
    return float(row[field])


def _normalize_training_hours(hours: float) -> float:
    return min(100.0, max(0.0, float(hours) / 100.0 * 100.0))


def calculate_role_match_score(row: pd.Series) -> float:
    model = ROLE_MODELS.get(row["target_role"], ROLE_MODELS["技术负责人"])
    score = sum(_field_score(row, field) * weight for _, field, weight in model)
    return round(score, 1)


def calculate_explicit_performance_score(row: pd.Series) -> float:
    training_score = _normalize_training_hours(row["training_hours"])
    score = (
        float(row["kpi_score"]) * 0.45
        + float(row["exam_score"]) * 0.25
        + float(row["manager_score"]) * 0.20
        + training_score * 0.10
    )
    return round(score, 1)


def calculate_hidden_contribution_score(row: pd.Series) -> float:
    score = (
        float(row["cross_functional_collaboration"]) * 0.20
        + float(row["technical_problem_solving"]) * 0.15
        + float(row["mentoring"]) * 0.15
        + float(row["knowledge_sharing"]) * 0.15
        + float(row["conflict_resolution"]) * 0.10
        + float(row["organizational_judgment"]) * 0.10
        + float(row["employee_relations"]) * 0.075
        + float(row["business_support"]) * 0.075
    )
    return round(score, 1)


def calculate_development_potential_score(row: pd.Series) -> float:
    training_score = _normalize_training_hours(row["training_hours"])
    score = (
        float(row["manager_score"]) * 0.25
        + float(row["peer_score"]) * 0.20
        + float(row["exam_score"]) * 0.15
        + training_score * 0.15
        + float(row["management_evidence"]) * 0.25
    )
    if bool(row["is_high_potential"]):
        score += 3
    return round(min(100.0, score), 1)


def calculate_total_score(
    explicit_performance: float,
    role_match: float,
    hidden_contribution: float,
    development_potential: float,
) -> float:
    score = (
        explicit_performance * 0.25
        + role_match * 0.40
        + hidden_contribution * 0.20
        + development_potential * 0.15
    )
    return round(score, 1)


def assess_risk_level(row: pd.Series, role_match: float, total_score: float) -> tuple[str, list[str]]:
    points = 0
    notes: list[str] = []

    if role_match < 60:
        points += 2
        notes.append("岗位匹配度明显不足")
    elif role_match < 70:
        points += 1
        notes.append("岗位匹配度处于观察区间")

    if float(row["kpi_score"]) > 85 and role_match < 65:
        points += 2
        notes.append("高绩效与目标岗位能力证据不匹配")

    if row["target_role"] == "团队管理者" and float(row["management_evidence"]) < 60:
        points += 2
        notes.append("管理能力证据不足")

    if row["target_role"] == "团队管理者" and float(row["team_satisfaction"]) < 60:
        points += 1
        notes.append("团队满意度偏低")

    if abs(total_score - float(row["manager_score"])) > 20:
        points += 1
        notes.append("AI 辅助评分与经理评分差异较大")

    if float(row["fairness_score"]) < 65:
        points += 1
        notes.append("公平感评分偏低")

    if total_score < 65:
        points += 1
        notes.append("总评分低于晋升观察线")

    if points >= 4:
        return "高风险", notes
    if points >= 2:
        return "中风险", notes
    return "低风险", notes or ["未发现明显风险信号"]


def identify_base_review_triggers(
    row: pd.Series,
    total_score: float,
    role_match: float,
    hidden_contribution: float,
) -> list[str]:
    triggers: list[str] = []

    if abs(total_score - float(row["manager_score"])) > 20:
        triggers.append("AI 总评分与经理评分差异超过 20 分")
    if hidden_contribution > 85 and total_score < 75:
        triggers.append("隐性贡献分高于 85，但总评分低于 75")
    if float(row["kpi_score"]) > 85 and role_match < 65:
        triggers.append("KPI 分数高于 85，但岗位匹配分低于 65")
    if row["target_role"] == "团队管理者" and float(row["management_evidence"]) < 60:
        triggers.append("申请团队管理者岗位，但管理能力证据分低于 60")

    return triggers


def determine_recommendation(
    row: pd.Series,
    total_score: float,
    risk_level: str,
    base_triggers: list[str],
) -> tuple[str, list[str]]:
    triggers = list(base_triggers)

    provisional_promotion = total_score >= 80 and risk_level == "低风险" and not base_triggers
    if bool(row["is_high_potential"]) and not provisional_promotion:
        triggers.append("高潜人才暂未被推荐晋升")

    if triggers:
        return "进入人工复核", triggers
    if total_score >= 80 and risk_level == "低风险":
        return "建议晋升", triggers
    if 65 <= total_score < 80:
        triggers.append("总评分位于 65-79 分灰区")
        return "进入人工复核", triggers
    if risk_level != "低风险":
        triggers.append("风险等级不是低风险")
        return "进入人工复核", triggers
    return "暂缓晋升并制定发展计划", triggers


def suggest_hr_actions(triggers: list[str], recommendation: str) -> str:
    actions = [ACTION_MAP[trigger] for trigger in triggers if trigger in ACTION_MAP]

    if recommendation == "建议晋升":
        actions.append("进入晋升委员会讨论")
    elif recommendation == "暂缓晋升并制定发展计划":
        actions.append("制定发展计划")

    unique_actions = list(dict.fromkeys(actions))
    return "；".join(unique_actions) if unique_actions else "安排 HRBP 访谈"


def score_employee(row: pd.Series) -> dict:
    explicit_performance = calculate_explicit_performance_score(row)
    role_match = calculate_role_match_score(row)
    hidden_contribution = calculate_hidden_contribution_score(row)
    development_potential = calculate_development_potential_score(row)
    total_score = calculate_total_score(
        explicit_performance,
        role_match,
        hidden_contribution,
        development_potential,
    )
    risk_level, risk_notes = assess_risk_level(row, role_match, total_score)
    base_triggers = identify_base_review_triggers(row, total_score, role_match, hidden_contribution)
    recommendation, triggers = determine_recommendation(row, total_score, risk_level, base_triggers)
    predicted_competency_rate = round(min(96.0, max(45.0, total_score * 0.86 + role_match * 0.14)), 1)
    high_risk_role_mismatch = (
        (float(row["kpi_score"]) > 85 and role_match < 65)
        or (row["target_role"] == "团队管理者" and float(row["management_evidence"]) < 60)
    )

    return {
        "explicit_performance_score": explicit_performance,
        "role_match_score": role_match,
        "hidden_contribution_score": hidden_contribution,
        "development_potential_score": development_potential,
        "total_score": total_score,
        "promotion_risk_level": risk_level,
        "risk_notes": "；".join(risk_notes),
        "ai_recommendation": recommendation,
        "review_triggers": "；".join(triggers) if triggers else "未触发人工复核规则",
        "needs_human_review": recommendation == "进入人工复核",
        "hr_action": suggest_hr_actions(triggers, recommendation),
        "predicted_competency_rate": predicted_competency_rate,
        "high_risk_role_mismatch": high_risk_role_mismatch,
    }


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    scored_rows = df.apply(score_employee, axis=1, result_type="expand")
    result = pd.concat([df.copy(), scored_rows], axis=1)
    for column in SCORE_COLUMNS + ["predicted_competency_rate"]:
        result[column] = result[column].round(1)
    return result
