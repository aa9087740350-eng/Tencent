"""Generate explainable HR-style feedback reports."""

from __future__ import annotations

import pandas as pd


def _yes_no(value: bool) -> str:
    return "是" if bool(value) else "否"


def _top_strengths(row: pd.Series) -> list[str]:
    candidates = [
        ("KPI 完成情况稳定", row["kpi_score"]),
        ("跨部门协作表现突出", row["cross_functional_collaboration"]),
        ("技术攻关能力较强", row["technical_problem_solving"]),
        ("带教新人和知识沉淀较好", (row["mentoring"] + row["knowledge_sharing"]) / 2),
        ("员工关系和业务支持能力较好", (row["employee_relations"] + row["business_support"]) / 2),
        ("组织判断和冲突处理能力较好", (row["organizational_judgment"] + row["conflict_resolution"]) / 2),
    ]
    strengths = [name for name, score in sorted(candidates, key=lambda item: item[1], reverse=True)[:3] if score >= 75]
    return strengths or ["当前表现较为均衡，但尚未形成特别突出的晋升证据"]


def _weaknesses(row: pd.Series) -> list[str]:
    weaknesses: list[str] = []
    if row["role_match_score"] < 70:
        weaknesses.append("目标岗位匹配证据仍需加强")
    if row["target_role"] == "团队管理者" and row["management_evidence"] < 65:
        weaknesses.append("管理能力、团队带动和组织判断证据仍不充分")
    if row["team_satisfaction"] < 65:
        weaknesses.append("团队满意度偏低，需要进一步核实团队协作体验")
    if row["hidden_contribution_score"] < 70:
        weaknesses.append("隐性贡献材料沉淀不足，可能影响评审完整性")
    if row["fairness_score"] < 65:
        weaknesses.append("公平感评分偏低，建议关注员工对评审过程的理解和信任")
    return weaknesses or ["暂未发现明显短板，建议继续补充关键项目证据"]


def _possibly_undervalued(row: pd.Series) -> str:
    if row["hidden_contribution_score"] >= 85 and row["total_score"] < 80:
        return "该员工的跨部门协作、带教新人或项目推动类贡献可能被纸面绩效低估，建议补充项目证据和协作方反馈。"
    if row["cross_functional_collaboration"] >= 85 or row["knowledge_sharing"] >= 85:
        return "该员工在协作推动、知识沉淀或团队支持方面可能存在被低估的组织贡献。"
    return "当前未发现明显被低估的贡献信号，但仍建议结合项目记录和 360 度反馈进行校验。"


def generate_feedback_report(row: pd.Series) -> str:
    strengths = "；".join(_top_strengths(row))
    weaknesses = "；".join(_weaknesses(row))
    undervalued = _possibly_undervalued(row)
    appeal_suggestion = (
        "建议提交补充材料或进入申诉 / 校准流程。"
        if row["ai_recommendation"] == "进入人工复核" or row["fairness_score"] < 65
        else "暂不需要提交申诉材料，但可继续补充关键项目证据。"
    )

    return f"""
### {row["name"]} 晋升评估反馈报告

**一、本次 AI 辅助评估结果**  
本次系统给出的 AI 辅助建议为：**{row["ai_recommendation"]}**。该结果用于帮助 HR 识别风险、整理证据和触发复核，不代表最终晋升决定。

**二、员工基本信息**  
- 部门：{row["department"]}  
- 当前岗位：{row["current_role"]}  
- 目标晋升岗位：{row["target_role"]}  
- 是否高潜员工：{_yes_no(row["is_high_potential"])}

**三、主要优势**  
{strengths}。

**四、当前不足**  
{weaknesses}。

**五、为什么给出该建议**  
该员工本次总评分为 **{row["total_score"]}**，岗位匹配分为 **{row["role_match_score"]}**，隐性贡献分为 **{row["hidden_contribution_score"]}**，发展潜力分为 **{row["development_potential_score"]}**。风险等级为 **{row["promotion_risk_level"]}**。复核触发原因：{row["review_triggers"]}。

**六、哪些贡献可能被低估**  
{undervalued}

**七、下一步发展建议**  
建议 HR 采取：{row["hr_action"]}。同时结合直属经理校准、项目贡献材料、协作方反馈和 360 度反馈，形成更完整的晋升判断。

**八、是否建议提交补充材料或申诉**  
{appeal_suggestion}
"""
