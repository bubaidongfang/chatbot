import streamlit as st
import requests
import json
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import time
from openai import OpenAI

# 设置页面配置
st.set_page_config(
    page_title="MacroInsight - 宏观新闻分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置第三方API配置
OPENAI_API_KEY = "xxxx"  # 更新为自己在tu-zi中的API
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.tu-zi.com/v1")

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #BFDBFE;
        padding-bottom: 0.3rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    .impact-high {
        color: #DC2626;
        font-weight: 600;
    }
    .impact-medium {
        color: #F59E0B;
        font-weight: 600;
    }
    .impact-low {
        color: #10B981;
        font-weight: 600;
    }
    .positive {
        color: #10B981;
        font-weight: 600;
    }
    .negative {
        color: #DC2626;
        font-weight: 600;
    }
    .neutral {
        color: #6B7280;
        font-weight: 600;
    }
    .highlight {
        background-color: #FEFCE8;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: 500;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏设置
with st.sidebar:
    st.markdown("## 分析选项")
    detailed_analysis = st.checkbox("详细分析", value=True, help="启用更深入的分析")
    include_charts = st.checkbox("包含可视化图表", value=True, help="生成影响可视化图表")

    st.markdown("## 模型信息")
    st.info("本工具使用 deepseek-reasoner 模型进行分析")

    st.markdown("## 关于")
    st.info(
        "MacroInsight是一款专业的宏观新闻分析工具，"
        "利用先进AI模型分析宏观经济新闻对金融市场的潜在影响。"
        "\n\n该工具提供对影响时长、方向、力度和消退指标的专业评估。"
    )

    st.markdown("### 使用指南")
    st.markdown(
        "1. 粘贴宏观经济或金融新闻内容\n"
        "2. 选择分析选项\n"
        "3. 点击'分析新闻'按钮\n"
        "4. 查看详细的多维度分析结果"
    )

    st.markdown("### API状态")
    try:
        # 简单测试API连接
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        st.success("✅ API连接正常")
    except Exception as e:
        st.error(f"❌ API连接异常: {str(e)}")

# 主页面
st.markdown('<div class="main-header">MacroInsight 宏观新闻分析工具</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">通过AI深度分析宏观经济新闻对金融市场的潜在影响，为投资决策提供专业参考</div>',
            unsafe_allow_html=True)

# 新闻输入区域
news_text = st.text_area(
    "输入宏观经济/金融新闻内容",
    height=200,
    help="粘贴完整的新闻文本，包括标题和正文内容"
)

# 分析按钮
if st.button("分析新闻", disabled=not news_text):
    if not news_text:
        st.error("请输入新闻内容")
    else:
        with st.spinner("正在分析新闻内容，这可能需要15-30秒..."):
            try:
                # 构建优化后的提示词
                prompt = f"""
                请对以下宏观经济/金融新闻进行全面分析：

                "{news_text}"

                请从以下四个维度进行专业评估：

                1. 影响时长：
                   - 该新闻对市场的影响周期是长期(数月至数年)、中期(数周至数月)还是短期(数小时至数天)?
                   - 基于哪些经济/金融理论或历史先例支持您的判断?
                   - 影响时长可能受哪些额外因素调节?

                2. 影响方向与范围：
                   - 对不同资产类别的影响方向(利好/利空/中性):
                     * 股票市场(细分行业影响差异)
                     * 债券市场(关注收益率曲线变化)
                     * 商品市场(能源、贵金属、农产品等)
                     * 外汇市场(主要货币对)
                     * 加密货币市场
                   - 是否存在跨市场传导效应?

                3. 影响力度评估：
                   - 预期市场波动幅度(高/中/低)及量化估计
                   - 与历史相似事件的对比分析
                   - 当前市场环境下的敏感性因素
                   - 可能的超预期反应情景

                4. 影响消退指标：
                   - 关键监测指标:
                     * 技术指标(价格、交易量、波动率等)
                     * 情绪指标(VIX、情绪指数、资金流向等)
                     * 关联市场反应
                   - 建议监测时间窗口
                   - 潜在的二次效应或长尾风险

                请提供基于数据和理论支持的分析，避免主观臆断，并在适当情况下指出分析的不确定性。

                请以JSON格式返回结果，格式如下：
                {{
                    "summary": "简要总结新闻内容和总体影响",
                    "duration": {{
                        "assessment": "长期/中期/短期",
                        "explanation": "详细解释",
                        "theory_support": "理论支持",
                        "modulating_factors": "可能的调节因素"
                    }},
                    "direction": {{
                        "stocks": {{
                            "overall": "利好/利空/中性",
                            "sectors": [
                                {{"sector": "行业1", "impact": "利好/利空/中性", "reason": "原因"}},
                                {{"sector": "行业2", "impact": "利好/利空/中性", "reason": "原因"}}
                            ]
                        }},
                        "bonds": {{
                            "overall": "利好/利空/中性",
                            "details": "详细解释",
                            "yield_curve": "收益率曲线变化"
                        }},
                        "commodities": {{
                            "overall": "利好/利空/中性",
                            "details": [
                                {{"type": "能源", "impact": "利好/利空/中性", "reason": "原因"}},
                                {{"type": "贵金属", "impact": "利好/利空/中性", "reason": "原因"}},
                                {{"type": "农产品", "impact": "利好/利空/中性", "reason": "原因"}}
                            ]
                        }},
                        "forex": {{
                            "overall": "详细说明",
                            "pairs": [
                                {{"pair": "货币对1", "impact": "升值/贬值", "magnitude": "幅度"}},
                                {{"pair": "货币对2", "impact": "升值/贬值", "magnitude": "幅度"}}
                            ]
                        }},
                        "crypto": {{
                            "overall": "利好/利空/中性",
                            "details": "详细解释",
                            "specific_coins": [
                                {{"coin": "比特币", "impact": "利好/利空/中性", "reason": "原因"}},
                                {{"coin": "以太坊", "impact": "利好/利空/中性", "reason": "原因"}}
                            ]
                        }},
                        "cross_market": "跨市场传导效应分析"
                    }},
                    "magnitude": {{
                        "overall": "高/中/低",
                        "quantitative": "量化估计",
                        "historical_comparison": "历史对比",
                        "sensitivity_factors": "敏感性因素",
                        "surprise_scenarios": "超预期情景"
                    }},
                    "monitoring": {{
                        "technical_indicators": [
                            {{"indicator": "指标1", "threshold": "阈值", "interpretation": "解释"}},
                            {{"indicator": "指标2", "threshold": "阈值", "interpretation": "解释"}}
                        ],
                        "sentiment_indicators": [
                            {{"indicator": "指标1", "threshold": "阈值", "interpretation": "解释"}},
                            {{"indicator": "指标2", "threshold": "阈值", "interpretation": "解释"}}
                        ],
                        "related_markets": [
                            {{"market": "市场1", "signal": "信号", "interpretation": "解释"}},
                            {{"market": "市场2", "signal": "信号", "interpretation": "解释"}}
                        ],
                        "monitoring_window": "建议监测时间窗口",
                        "secondary_effects": "二次效应分析",
                        "tail_risks": "长尾风险"
                    }}
                }}
                """

                # 使用OpenAI客户端调用API，指定使用deepseek-reasoner模型
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=4000
                )

                analysis_text = response.choices[0].message.content

                # 提取JSON部分
                try:
                    # 查找JSON开始和结束的位置
                    start_idx = analysis_text.find('{')
                    end_idx = analysis_text.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = analysis_text[start_idx:end_idx]
                        analysis = json.loads(json_str)
                    else:
                        # 如果找不到完整的JSON，尝试使用整个响应
                        analysis = json.loads(analysis_text)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，显示原始文本
                    st.error("无法解析API返回的JSON数据，显示原始分析结果：")
                    st.write(analysis_text)
                    st.stop()

                # 显示分析结果
                st.markdown('<div class="sub-header">新闻摘要与整体影响</div>', unsafe_allow_html=True)
                st.markdown(f"<div class='info-box'>{analysis['summary']}</div>", unsafe_allow_html=True)

                # 影响时长
                st.markdown('<div class="sub-header">1. 影响时长分析</div>', unsafe_allow_html=True)
                duration_class = ""
                if analysis['duration']['assessment'].lower() == "长期":
                    duration_class = "impact-high"
                elif analysis['duration']['assessment'].lower() == "中期":
                    duration_class = "impact-medium"
                else:
                    duration_class = "impact-low"

                st.markdown(
                    f"**评估结果：** <span class='{duration_class}'>{analysis['duration']['assessment']}</span>",
                    unsafe_allow_html=True)
                st.markdown(f"**详细解释：** {analysis['duration']['explanation']}")
                st.markdown(f"**理论支持：** {analysis['duration']['theory_support']}")
                st.markdown(f"**调节因素：** {analysis['duration']['modulating_factors']}")

                # 影响方向
                st.markdown('<div class="sub-header">2. 影响方向与范围</div>', unsafe_allow_html=True)

                # 创建资产类别影响表格
                impact_data = {
                    "资产类别": ["股票市场", "债券市场", "商品市场", "外汇市场", "加密货币市场"],
                    "整体影响": [
                        analysis['direction']['stocks']['overall'],
                        analysis['direction']['bonds']['overall'],
                        analysis['direction']['commodities']['overall'],
                        "详见具体货币对",
                        analysis['direction']['crypto']['overall']
                    ]
                }

                impact_df = pd.DataFrame(impact_data)

                # 使用自定义样式显示表格
                st.markdown("#### 各资产类别整体影响")


                # 自定义表格显示
                def color_impact(val):
                    if val.lower() in ["利好", "正面", "积极"]:
                        return 'background-color: #DCFCE7; color: #166534'
                    elif val.lower() in ["利空", "负面", "消极"]:
                        return 'background-color: #FEE2E2; color: #991B1B'
                    elif val.lower() in ["中性", "混合"]:
                        return 'background-color: #F3F4F6; color: #4B5563'
                    else:
                        return ''


                st.dataframe(impact_df.style.map(color_impact, subset=['整体影响']))

                # 股票市场详细分析
                st.markdown("#### 股票市场行业影响")
                if 'sectors' in analysis['direction']['stocks']:
                    sectors_data = []
                    for sector in analysis['direction']['stocks']['sectors']:
                        sectors_data.append([
                            sector['sector'],
                            sector['impact'],
                            sector['reason']
                        ])

                    sectors_df = pd.DataFrame(sectors_data, columns=["行业", "影响", "原因"])
                    st.dataframe(sectors_df.style.map(color_impact, subset=['影响']))
                else:
                    st.write("无行业详细分析")

                # 债券市场
                st.markdown("#### 债券市场详细分析")
                st.markdown(f"**收益率曲线变化：** {analysis['direction']['bonds']['yield_curve']}")
                st.markdown(f"**详细解释：** {analysis['direction']['bonds']['details']}")

                # 商品市场
                st.markdown("#### 商品市场详细分析")
                if 'details' in analysis['direction']['commodities']:
                    commodities_data = []
                    for commodity in analysis['direction']['commodities']['details']:
                        commodities_data.append([
                            commodity['type'],
                            commodity['impact'],
                            commodity['reason']
                        ])

                    commodities_df = pd.DataFrame(commodities_data, columns=["商品类型", "影响", "原因"])
                    st.dataframe(commodities_df.style.map(color_impact, subset=['影响']))
                else:
                    st.write("无商品详细分析")

                # 外汇市场
                st.markdown("#### 外汇市场详细分析")
                if 'pairs' in analysis['direction']['forex']:
                    forex_data = []
                    for pair in analysis['direction']['forex']['pairs']:
                        forex_data.append([
                            pair['pair'],
                            pair['impact'],
                            pair['magnitude']
                        ])

                    forex_df = pd.DataFrame(forex_data, columns=["货币对", "影响方向", "预期幅度"])
                    st.dataframe(forex_df)
                else:
                    st.write("无外汇详细分析")

                # 加密货币
                st.markdown("#### 加密货币详细分析")
                st.markdown(f"**整体评估：** {analysis['direction']['crypto']['details']}")

                if 'specific_coins' in analysis['direction']['crypto']:
                    crypto_data = []
                    for coin in analysis['direction']['crypto']['specific_coins']:
                        crypto_data.append([
                            coin['coin'],
                            coin['impact'],
                            coin['reason']
                        ])

                    crypto_df = pd.DataFrame(crypto_data, columns=["加密货币", "影响", "原因"])
                    st.dataframe(crypto_df.style.map(color_impact, subset=['影响']))
                else:
                    st.write("无加密货币详细分析")

                # 跨市场传导效应
                st.markdown("#### 跨市场传导效应")
                st.markdown(analysis['direction']['cross_market'])

                # 影响力度
                st.markdown('<div class="sub-header">3. 影响力度评估</div>', unsafe_allow_html=True)

                magnitude_class = ""
                if analysis['magnitude']['overall'].lower() == "高":
                    magnitude_class = "impact-high"
                elif analysis['magnitude']['overall'].lower() == "中":
                    magnitude_class = "impact-medium"
                else:
                    magnitude_class = "impact-low"

                st.markdown(
                    f"**整体评估：** <span class='{magnitude_class}'>{analysis['magnitude']['overall']}</span>",
                    unsafe_allow_html=True)
                st.markdown(f"**量化估计：** {analysis['magnitude']['quantitative']}")
                st.markdown(f"**历史对比：** {analysis['magnitude']['historical_comparison']}")
                st.markdown(f"**敏感性因素：** {analysis['magnitude']['sensitivity_factors']}")
                st.markdown(f"**超预期情景：** {analysis['magnitude']['surprise_scenarios']}")

                # 可视化影响力度
                if include_charts:
                    st.markdown("#### 影响力度可视化")

                    # 创建资产类别影响力度的仪表盘图
                    impact_levels = {
                        "股票市场": 0,
                        "债券市场": 0,
                        "商品市场": 0,
                        "外汇市场": 0,
                        "加密货币市场": 0
                    }

                    # 根据分析结果设置影响力度值
                    for asset, impact in zip(impact_data["资产类别"], impact_data["整体影响"]):
                        if impact.lower() in ["强烈利好", "显著利好", "强利好"]:
                            impact_levels[asset] = 0.9
                        elif impact.lower() in ["利好", "正面", "积极"]:
                            impact_levels[asset] = 0.7
                        elif impact.lower() in ["轻微利好", "小幅利好"]:
                            impact_levels[asset] = 0.6
                        elif impact.lower() in ["中性", "混合"]:
                            impact_levels[asset] = 0.5
                        elif impact.lower() in ["轻微利空", "小幅利空"]:
                            impact_levels[asset] = 0.4
                        elif impact.lower() in ["利空", "负面", "消极"]:
                            impact_levels[asset] = 0.3
                        elif impact.lower() in ["强烈利空", "显著利空", "强利空"]:
                            impact_levels[asset] = 0.1
                        else:
                            impact_levels[asset] = 0.5

                    # 创建雷达图
                    categories = list(impact_levels.keys())
                    values = list(impact_levels.values())

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='影响力度',
                        line_color='rgb(38, 139, 210)',
                        fillcolor='rgba(38, 139, 210, 0.3)'
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickvals=[0, 0.25, 0.5, 0.75, 1],
                                ticktext=['强烈利空', '利空', '中性', '利好', '强烈利好']
                            )
                        ),
                        showlegend=False,
                        title="各资产类别影响力度雷达图",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # 影响消退指标
                st.markdown('<div class="sub-header">4. 影响消退指标</div>', unsafe_allow_html=True)

                # 技术指标
                st.markdown("#### 技术指标监测")
                if 'technical_indicators' in analysis['monitoring']:
                    tech_data = []
                    for indicator in analysis['monitoring']['technical_indicators']:
                        tech_data.append([
                            indicator['indicator'],
                            indicator['threshold'],
                            indicator['interpretation']
                        ])

                    tech_df = pd.DataFrame(tech_data, columns=["指标", "阈值", "解释"])
                    st.dataframe(tech_df)
                else:
                    st.write("无技术指标详细分析")

                # 情绪指标
                st.markdown("#### 情绪指标监测")
                if 'sentiment_indicators' in analysis['monitoring']:
                    sentiment_data = []
                    for indicator in analysis['monitoring']['sentiment_indicators']:
                        sentiment_data.append([
                            indicator['indicator'],
                            indicator['threshold'],
                            indicator['interpretation']
                        ])

                    sentiment_df = pd.DataFrame(sentiment_data, columns=["指标", "阈值", "解释"])
                    st.dataframe(sentiment_df)
                else:
                    st.write("无情绪指标详细分析")

                # 关联市场
                st.markdown("#### 关联市场监测")
                if 'related_markets' in analysis['monitoring']:
                    related_data = []
                    for market in analysis['monitoring']['related_markets']:
                        related_data.append([
                            market['market'],
                            market['signal'],
                            market['interpretation']
                        ])

                    related_df = pd.DataFrame(related_data, columns=["市场", "信号", "解释"])
                    st.dataframe(related_df)
                else:
                    st.write("无关联市场详细分析")

                st.markdown(f"**建议监测时间窗口：** {analysis['monitoring']['monitoring_window']}")
                st.markdown(f"**二次效应分析：** {analysis['monitoring']['secondary_effects']}")
                st.markdown(f"**长尾风险：** {analysis['monitoring']['tail_risks']}")

                # 添加总结和建议部分
                st.markdown('<div class="sub-header">总结与投资建议</div>', unsafe_allow_html=True)

                # 生成总结建议
                summary_prompt = f"""
                基于以下宏观新闻分析，提供简明的投资建议总结：

                新闻内容: {news_text}

                影响时长: {analysis['duration']['assessment']}
                整体影响力度: {analysis['magnitude']['overall']}

                股票市场影响: {analysis['direction']['stocks']['overall']}
                债券市场影响: {analysis['direction']['bonds']['overall']}
                商品市场影响: {analysis['direction']['commodities']['overall']}
                加密货币市场影响: {analysis['direction']['crypto']['overall']}

                请提供200字以内的投资建议总结，包括风险提示。
                """

                # 调用API获取投资建议，同样使用deepseek-reasoner模型
                summary_response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.7,
                    max_tokens=500
                )

                investment_advice = summary_response.choices[0].message.content
                st.markdown(f'<div class="info-box">{investment_advice}</div>', unsafe_allow_html=True)

                # 导出报告选项
                st.markdown('<div class="sub-header">导出分析报告</div>', unsafe_allow_html=True)

                # 生成完整报告文本
                report_text = f"""
                # 宏观新闻影响分析报告

                ## 分析新闻
                {news_text}

                ## 新闻摘要与整体影响
                {analysis['summary']}

                ## 1. 影响时长分析
                - **评估结果：** {analysis['duration']['assessment']}
                - **详细解释：** {analysis['duration']['explanation']}
                - **理论支持：** {analysis['duration']['theory_support']}
                - **调节因素：** {analysis['duration']['modulating_factors']}

                ## 2. 影响方向与范围

                ### 股票市场
                - **整体影响：** {analysis['direction']['stocks']['overall']}
                - **行业影响：** 
                """

                # 添加行业影响
                if 'sectors' in analysis['direction']['stocks']:
                    for sector in analysis['direction']['stocks']['sectors']:
                        report_text += f"\n  - {sector['sector']}: {sector['impact']} - {sector['reason']}"

                report_text += f"""

                ### 债券市场
                - **整体影响：** {analysis['direction']['bonds']['overall']}
                - **收益率曲线变化：** {analysis['direction']['bonds']['yield_curve']}
                - **详细解释：** {analysis['direction']['bonds']['details']}

                ### 商品市场
                - **整体影响：** {analysis['direction']['commodities']['overall']}
                """

                # 添加商品详情
                if 'details' in analysis['direction']['commodities']:
                    for commodity in analysis['direction']['commodities']['details']:
                        report_text += f"\n  - {commodity['type']}: {commodity['impact']} - {commodity['reason']}"

                report_text += f"""

                ### 外汇市场
                - **整体影响：** {analysis['direction']['forex']['overall']}
                """

                # 添加外汇详情
                if 'pairs' in analysis['direction']['forex']:
                    for pair in analysis['direction']['forex']['pairs']:
                        report_text += f"\n  - {pair['pair']}: {pair['impact']} ({pair['magnitude']})"

                report_text += f"""

                ### 加密货币市场
                - **整体影响：** {analysis['direction']['crypto']['overall']}
                - **详细解释：** {analysis['direction']['crypto']['details']}
                """

                # 添加加密货币详情
                if 'specific_coins' in analysis['direction']['crypto']:
                    for coin in analysis['direction']['crypto']['specific_coins']:
                        report_text += f"\n  - {coin['coin']}: {coin['impact']} - {coin['reason']}"

                report_text += f"""

                ### 跨市场传导效应
                {analysis['direction']['cross_market']}

                ## 3. 影响力度评估
                - **整体评估：** {analysis['magnitude']['overall']}
                - **量化估计：** {analysis['magnitude']['quantitative']}
                - **历史对比：** {analysis['magnitude']['historical_comparison']}
                - **敏感性因素：** {analysis['magnitude']['sensitivity_factors']}
                - **超预期情景：** {analysis['magnitude']['surprise_scenarios']}

                ## 4. 影响消退指标

                ### 技术指标监测
                """

                # 添加技术指标
                if 'technical_indicators' in analysis['monitoring']:
                    for indicator in analysis['monitoring']['technical_indicators']:
                        report_text += f"\n  - {indicator['indicator']}: {indicator['threshold']} - {indicator['interpretation']}"

                report_text += f"""

                ### 情绪指标监测
                """

                # 添加情绪指标
                if 'sentiment_indicators' in analysis['monitoring']:
                    for indicator in analysis['monitoring']['sentiment_indicators']:
                        report_text += f"\n  - {indicator['indicator']}: {indicator['threshold']} - {indicator['interpretation']}"

                report_text += f"""

                ### 关联市场监测
                """

                # 添加关联市场
                if 'related_markets' in analysis['monitoring']:
                    for market in analysis['monitoring']['related_markets']:
                        report_text += f"\n  - {market['market']}: {market['signal']} - {market['interpretation']}"

                report_text += f"""

                - **建议监测时间窗口：** {analysis['monitoring']['monitoring_window']}
                - **二次效应分析：** {analysis['monitoring']['secondary_effects']}
                - **长尾风险：** {analysis['monitoring']['tail_risks']}

                ## 投资建议总结
                {investment_advice}

                ---
                *报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
                *由 MacroInsight 宏观新闻分析工具生成*
                """

                # 提供下载报告选项
                st.download_button(
                    label="下载完整分析报告 (Markdown)",
                    data=report_text,
                    file_name=f"宏观新闻分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")
                st.error("如果是API错误，请检查网络连接或稍后再试")

# 添加页脚
st.markdown('<div class="footer">© 2025 MacroInsight 宏观新闻分析工具 | 基于先进AI模型</div>',
            unsafe_allow_html=True)
