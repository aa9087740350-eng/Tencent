from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.feedback import generate_feedback_report
from utils.hidden_contribution import detect_hidden_contributions
from utils.scoring import apply_scoring


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_employees.csv"

RISK_COLOR_MAP = {
    "低风险": "#16a34a",
    "中风险": "#f59e0b",
    "高风险": "#dc2626",
}

RECOMMENDATION_COLOR_MAP = {
    "建议晋升": "#16a34a",
    "进入人工复核": "#2563eb",
    "暂缓晋升并制定发展计划": "#f59e0b",
}


st.set_page_config(
    page_title="TalentLens AI 智能晋升评估助手",
    page_icon="T",
    layout="wide",
)


@st.cache_data
def load_employee_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    return apply_scoring(data)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tl-blue: #2563eb;
            --tl-navy: #172033;
            --tl-muted: #64748b;
            --tl-border: #e2e8f0;
            --tl-bg: #f6f8fb;
        }
        .stApp {
            background: var(--tl-bg);
            color: var(--tl-navy);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.6rem;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--tl-border);
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--tl-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--tl-muted);
            font-size: 0.86rem;
        }
        .section-note {
            background: #ffffff;
            border-left: 4px solid var(--tl-blue);
            border-radius: 6px;
            padding: 12px 16px;
            color: #334155;
            margin: 0.4rem 0 1.2rem 0;
        }
        .info-row {
            background: #ffffff;
            border: 1px solid var(--tl-border);
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 74px;
        }
        .info-label {
            color: var(--tl-muted);
            font-size: 0.82rem;
            margin-bottom: 6px;
        }
        .info-value {
            color: var(--tl-navy);
            font-weight: 650;
            line-height: 1.25;
        }
        .tag {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 4px 10px;
            margin: 4px 6px 4px 0;
            font-size: 0.88rem;
        }
        .footer {
            margin-top: 2.2rem;
            padding: 14px 0 0 0;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.92rem;
        }
        h1, h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown('<div class="footer">AI 辅助 HR 决策，但不替代人工判断</div>', unsafe_allow_html=True)


def render_info_box(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="info-row">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    color = RISK_COLOR_MAP.get(risk_level, "#64748b")
    return f'<span style="color:{color};font-weight:700;">{risk_level}</span>'


def recommendation_message(recommendation: str) -> None:
    if recommendation == "建议晋升":
        st.success("AI 辅助建议：建议晋升。请 HR 结合晋升委员会和管理者判断完成最终确认。")
    elif recommendation == "进入人工复核":
        st.info("AI 辅助建议：进入人工复核。建议补充证据、开展校准或收集 360 度反馈。")
    else:
        st.warning("AI 辅助建议：暂缓晋升并制定发展计划。建议明确下一阶段能力建设目标。")


def page_dashboard(df: pd.DataFrame) -> None:
    st.title("晋升评估总览")
    st.markdown(
        '<div class="section-note">首页不是直接决定谁晋升，而是帮助 HR 识别本轮晋升评审中的风险部门、风险员工和异常案例。</div>',
        unsafe_allow_html=True,
    )

    total_people = len(df)
    promoted_people = int((df["ai_recommendation"] == "建议晋升").sum())
    review_people = int(df["needs_human_review"].sum())
    mismatch_cases = int(df["high_risk_role_mismatch"].sum())
    avg_fairness = df["fairness_score"].mean()
    predicted_rate = df["predicted_competency_rate"].mean()

    metric_cols = st.columns(6)
    metrics = [
        ("参与晋升人数", f"{total_people} 人"),
        ("AI 辅助推荐晋升人数", f"{promoted_people} 人"),
        ("需要人工复核人数", f"{review_people} 人"),
        ("高风险人岗错配案例", f"{mismatch_cases} 例"),
        ("平均公平感评分", f"{avg_fairness:.1f}"),
        ("预测岗位胜任率", f"{predicted_rate:.1f}%"),
    ]
    for column, (label, value) in zip(metric_cols, metrics):
        column.metric(label, value)

    st.divider()

    left, right = st.columns(2)

    department_summary = (
        df.groupby("department")
        .agg(参与人数=("employee_id", "count"), 推荐晋升人数=("ai_recommendation", lambda x: (x == "建议晋升").sum()))
        .reset_index()
    )
    department_summary["晋升推荐比例"] = department_summary["推荐晋升人数"] / department_summary["参与人数"] * 100
    department_fig = px.bar(
        department_summary.sort_values("晋升推荐比例", ascending=False),
        x="department",
        y="晋升推荐比例",
        text="晋升推荐比例",
        color="推荐晋升人数",
        color_continuous_scale=["#dbeafe", "#2563eb"],
        title="各部门晋升推荐比例",
        labels={"department": "部门", "晋升推荐比例": "推荐比例（%）", "推荐晋升人数": "推荐人数"},
    )
    department_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    department_fig.update_layout(showlegend=False, yaxis_range=[0, 105], margin=dict(t=60, b=40))
    left.plotly_chart(department_fig, use_container_width=True)

    risk_counts = (
        df["promotion_risk_level"]
        .value_counts()
        .reindex(["低风险", "中风险", "高风险"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["风险等级", "人数"]
    risk_fig = px.bar(
        risk_counts,
        x="风险等级",
        y="人数",
        color="风险等级",
        color_discrete_map=RISK_COLOR_MAP,
        title="晋升风险分布",
        text="人数",
    )
    risk_fig.update_layout(showlegend=False, margin=dict(t=60, b=40))
    right.plotly_chart(risk_fig, use_container_width=True)

    left, right = st.columns(2)
    dimension_df = pd.DataFrame(
        {
            "评分维度": ["显性绩效分", "岗位匹配分", "隐性贡献分", "发展潜力分", "总评分"],
            "平均分": [
                df["explicit_performance_score"].mean(),
                df["role_match_score"].mean(),
                df["hidden_contribution_score"].mean(),
                df["development_potential_score"].mean(),
                df["total_score"].mean(),
            ],
        }
    )
    dimension_fig = px.bar(
        dimension_df,
        x="评分维度",
        y="平均分",
        title="不同评分维度平均分",
        text="平均分",
        color="评分维度",
        color_discrete_sequence=["#2563eb", "#16a34a", "#7c3aed", "#f59e0b", "#0f766e"],
    )
    dimension_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    dimension_fig.update_layout(showlegend=False, yaxis_range=[0, 105], margin=dict(t=60, b=40))
    left.plotly_chart(dimension_fig, use_container_width=True)

    diff_df = df.assign(评分差异=df["total_score"] - df["manager_score"])
    diff_fig = px.histogram(
        diff_df,
        x="评分差异",
        nbins=10,
        title="AI 评分与人工评价差异分布",
        labels={"评分差异": "AI 总评分 - 经理评分"},
        color_discrete_sequence=["#2563eb"],
    )
    diff_fig.add_vline(x=20, line_dash="dash", line_color="#dc2626", annotation_text="差异 +20")
    diff_fig.add_vline(x=-20, line_dash="dash", line_color="#dc2626", annotation_text="差异 -20")
    diff_fig.update_layout(yaxis_title="人数", margin=dict(t=60, b=40))
    right.plotly_chart(diff_fig, use_container_width=True)


def page_employee_detail(df: pd.DataFrame) -> None:
    st.title("员工评估详情")
    selected_name = st.selectbox(
        "选择员工",
        df["name"].tolist(),
        format_func=lambda name: f"{name}｜{df.loc[df['name'] == name, 'department'].iloc[0]}",
    )
    row = df.loc[df["name"] == selected_name].iloc[0]

    st.subheader("员工基本信息")
    info_cols = st.columns(5)
    info_items = [
        ("姓名", row["name"]),
        ("部门", row["department"]),
        ("当前岗位", row["current_role"]),
        ("目标晋升岗位", row["target_role"]),
        ("是否高潜员工", "是" if row["is_high_potential"] else "否"),
    ]
    for column, (label, value) in zip(info_cols, info_items):
        with column:
            render_info_box(label, value)

    st.subheader("多维评分")
    score_cols = st.columns(6)
    score_items = [
        ("显性绩效分", row["explicit_performance_score"]),
        ("岗位匹配分", row["role_match_score"]),
        ("隐性贡献分", row["hidden_contribution_score"]),
        ("发展潜力分", row["development_potential_score"]),
        ("总评分", row["total_score"]),
        ("晋升风险等级", row["promotion_risk_level"]),
    ]
    for column, (label, value) in zip(score_cols, score_items):
        if label == "晋升风险等级":
            column.markdown(f"**{label}**  \n{risk_badge(value)}", unsafe_allow_html=True)
        else:
            column.metric(label, f"{value:.1f}")

    chart_col, note_col = st.columns([1.2, 1])
    with chart_col:
        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    row["explicit_performance_score"],
                    row["role_match_score"],
                    row["hidden_contribution_score"],
                    row["development_potential_score"],
                ],
                theta=["显性绩效分", "岗位匹配分", "隐性贡献分", "发展潜力分"],
                fill="toself",
                name=row["name"],
                line_color="#2563eb",
            )
        )
        radar_fig.update_layout(
            title="员工多维评分雷达图",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(t=60, b=30),
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with note_col:
        st.subheader("AI 辅助建议")
        recommendation_message(row["ai_recommendation"])
        st.markdown(f"**复核触发原因：** {row['review_triggers']}")
        st.markdown(f"**建议 HR 动作：** {row['hr_action']}")
        st.markdown(f"**风险说明：** {row['risk_notes']}")


def page_hidden_contribution() -> None:
    st.title("隐性贡献识别器")
    st.markdown(
        '<div class="section-note">这个模块用于解决 AI 只看纸面数据、忽视员工真实组织贡献的问题。识别结果仅作为人工复核线索。</div>',
        unsafe_allow_html=True,
    )

    default_text = "过去半年，我参与了支付系统稳定性优化项目，协调技术、产品、客服三个部门，解决了上线后的高频故障问题，并带教两名新人完成模块交接。"
    contribution_text = st.text_area("请输入贡献描述", value=default_text, height=180)
    submitted = st.button("识别隐性贡献", type="primary", use_container_width=True)

    if submitted or contribution_text.strip():
        result = detect_hidden_contributions(contribution_text)
        st.subheader("识别结果")

        if result["tags"]:
            tag_html = "".join(f'<span class="tag">{tag}</span>' for tag in result["tags"])
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.info("暂未识别到明确隐性贡献标签。")

        score_col, review_col = st.columns(2)
        score_col.metric("建议隐性贡献评分", f"{result['score']} / 100")
        review_col.metric("是否建议进入人工复核", "是" if result["needs_review"] else "否")
        st.progress(result["score"] / 100)
        st.markdown(f"**隐性贡献总结：** {result['summary']}")
        st.warning(result["validation_note"])


def page_review_pool(df: pd.DataFrame) -> None:
    st.title("人工复核池")
    st.markdown(
        '<div class="section-note">人工复核池的作用是防止算法误判直接变成晋升结果。HR 可以在这里优先处理异常评分、灰区评分和高潜人才未推荐案例。</div>',
        unsafe_allow_html=True,
    )

    department_options = ["全部部门"] + sorted(df["department"].unique().tolist())
    selected_department = st.selectbox("筛选部门", department_options)
    risk_options = ["全部风险等级", "低风险", "中风险", "高风险"]
    selected_risk = st.selectbox("筛选风险等级", risk_options)

    review_df = df[df["needs_human_review"]].copy()
    if selected_department != "全部部门":
        review_df = review_df[review_df["department"] == selected_department]
    if selected_risk != "全部风险等级":
        review_df = review_df[review_df["promotion_risk_level"] == selected_risk]

    st.metric("当前复核池人数", f"{len(review_df)} 人")

    display_df = review_df[
        [
            "name",
            "department",
            "target_role",
            "total_score",
            "manager_score",
            "hidden_contribution_score",
            "review_triggers",
            "hr_action",
        ]
    ].rename(
        columns={
            "name": "员工姓名",
            "department": "部门",
            "target_role": "目标晋升岗位",
            "total_score": "总评分",
            "manager_score": "经理评分",
            "hidden_contribution_score": "隐性贡献分",
            "review_triggers": "复核触发原因",
            "hr_action": "建议 HR 动作",
        }
    )
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    if not review_df.empty:
        action_summary = review_df["hr_action"].str.split("；").explode().value_counts().reset_index()
        action_summary.columns = ["建议 HR 动作", "人数"]
        action_fig = px.bar(
            action_summary,
            x="建议 HR 动作",
            y="人数",
            title="人工复核建议动作分布",
            text="人数",
            color="建议 HR 动作",
            color_discrete_sequence=["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0f766e", "#dc2626"],
        )
        action_fig.update_layout(showlegend=False, margin=dict(t=60, b=40))
        st.plotly_chart(action_fig, use_container_width=True)


def page_feedback_report(df: pd.DataFrame) -> None:
    st.title("可解释反馈报告")
    selected_name = st.selectbox(
        "选择员工",
        df["name"].tolist(),
        format_func=lambda name: f"{name}｜{df.loc[df['name'] == name, 'target_role'].iloc[0]}",
    )
    row = df.loc[df["name"] == selected_name].iloc[0]
    report = generate_feedback_report(row)

    st.markdown(report)
    st.download_button(
        "下载反馈报告",
        data=report,
        file_name=f"{row['employee_id']}_{row['name']}_晋升反馈报告.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main() -> None:
    inject_style()
    df = load_employee_data()

    st.sidebar.title("TalentLens AI")
    st.sidebar.caption("智能晋升评估助手")
    page = st.sidebar.radio(
        "导航",
        ["首页总览", "员工评估详情", "隐性贡献识别器", "人工复核池", "可解释反馈报告"],
    )
    st.sidebar.divider()
    st.sidebar.markdown("**评估原则**")
    st.sidebar.markdown("AI 负责数据整理、风险识别和解释生成；HR 负责人工复核、复杂判断和最终决策。")

    if page == "首页总览":
        page_dashboard(df)
    elif page == "员工评估详情":
        page_employee_detail(df)
    elif page == "隐性贡献识别器":
        page_hidden_contribution()
    elif page == "人工复核池":
        page_review_pool(df)
    else:
        page_feedback_report(df)

    render_footer()


if __name__ == "__main__":
    main()
