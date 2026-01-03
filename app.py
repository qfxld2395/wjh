import streamlit as st
import pandas as pd
import joblib
import os

# 全局变量：截图列表
SCREENSHOTS = [
    "1.jpg",
    "2.jpg",
    "3.jpg"
]

# 设置页面配置
st.set_page_config(
    page_title='学生成绩分析与预测系统',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 加载模型和数据
@st.cache_resource
def load_model():
    model = joblib.load('score_prediction_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('student_data_adjusted_rounded.csv')

model, features = load_model()
df = load_data()

# 侧边栏导航
with st.sidebar:
    # 标题和欢迎信息
    st.title('📊 学生成绩分析与预测系统')
    st.markdown("---")
    
    # 功能模块导航
    st.subheader("功能模块")
    page = st.radio(
        "选择功能模块",
        [
            "🏠 项目介绍",
            "📈 专业数据分析", 
            "🤖 期末成绩预测"
        ],
        index=0,
        label_visibility='collapsed'
    )
    
    # 提取实际页面名称（去除图标）
    page = page.split(' ')[1]
    
    st.markdown("---")
    st.info("💡 提示：使用左侧导航切换功能模块")
    
# 确保使用浅色模式并优化配色
os.makedirs('.streamlit', exist_ok=True)
with open('.streamlit/config.toml', 'w') as f:
    f.write('''[theme]
base = "light"
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#1e293b"
font = "sans serif"
''')

# 页面1：项目介绍
if page == '项目介绍':
    # 页面标题和副标题
    st.title('📊 学生成绩分析与预测系统')
    st.subheader('用数据驱动教育，用智能预测未来')
    st.divider()
    
    # 项目概述
    st.subheader('🔍 项目概述')
    
    # 现代化的概述布局
    with st.container():
        # 使用Streamlit原生布局展示概述
        col1, col2 = st.columns([1, 1.5], gap="large")
        
        with col1:
            with st.container(border=True):
                st.subheader('📈 关于系统')
                st.write("本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。")
                st.write("系统提供了专业的数据分析工具和直观的可视化图表，让数据说话，为教育决策提供支持。")
        
        with col2:
            # 截图展示区域
            st.subheader('🖼️ 系统截图')
            
            if len(SCREENSHOTS) > 1:
                # 初始化状态
                if 'current_screenshot' not in st.session_state:
                    st.session_state.current_screenshot = 0
                
                # 图片容器
                with st.container():
                    st.image(
                        SCREENSHOTS[st.session_state.current_screenshot], 
                        width="stretch",
                        caption=f"截图 {st.session_state.current_screenshot + 1}/{len(SCREENSHOTS)}"
                    )
                
                # 按钮容器
                btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
                with btn_col1:
                    if st.button("◀ 上一张", key="prev_btn"):
                        st.session_state.current_screenshot = (st.session_state.current_screenshot - 1) % len(SCREENSHOTS)
                        st.rerun()
                with btn_col3:
                    if st.button("下一张 ▶", key="next_btn"):
                        st.session_state.current_screenshot = (st.session_state.current_screenshot + 1) % len(SCREENSHOTS)
                        st.rerun()
            else:
                st.image(SCREENSHOTS[0], width="stretch", caption="系统截图")
    
    st.divider()
    
    # 主要特点 - 使用Streamlit原生卡片布局
    st.subheader('✨ 主要特点')
    
    feature_cards = st.columns(4, gap="medium")
    
    with feature_cards[0]:
        with st.container(border=True):
            st.markdown("📊")
            st.subheader("数据可视化")
            st.write("多维度展示学生学业数据，直观清晰")
    
    with feature_cards[1]:
        with st.container(border=True):
            st.markdown("📚")
            st.subheader("专业分析")
            st.write("按专业分类的详细统计分析，深入洞察")
    
    with feature_cards[2]:
        with st.container(border=True):
            st.markdown("🤖")
            st.subheader("智能预测")
            st.write("基于机器学习模型的成绩预测，准确可靠")
    
    with feature_cards[3]:
        with st.container(border=True):
            st.markdown("💡")
            st.subheader("学习建议")
            st.write("根据预测结果提供个性化反馈和建议")
    
    st.divider()
    
    # 项目目标
    st.subheader('🎯 项目目标')
    
    goal_cards = st.columns(3, gap="large")
    
    with goal_cards[0]:
        with st.container(border=True):
            st.subheader("🎯 目标一")
            st.write("- 实现学生成绩数据的可视化分析")
            st.write("- 提供多维度的数据统计")
            st.write("- 帮助教师了解教学效果")
    
    with goal_cards[1]:
        with st.container(border=True):
            st.subheader("🎯 目标二")
            st.write("- 建立准确的成绩预测模型")
            st.write("- 帮助学生了解自身学习情况")
            st.write("- 提供个性化的学习建议")
    
    with goal_cards[2]:
        with st.container(border=True):
            st.subheader("🎯 目标三")
            st.write("- 提升学生学习积极性")
            st.write("- 促进教学质量的提高")
            st.write("- 实现数据驱动的教学管理")
    
    st.divider()
    
    # 技术架构
    st.subheader('🛠️ 技术架构')
    
    tech_cards = st.columns(4, gap="medium")
    
    with tech_cards[0]:
        with st.container(border=True):
            st.markdown("🖥️")
            st.subheader("前端框架")
            st.write("Streamlit")
    
    with tech_cards[1]:
        with st.container(border=True):
            st.markdown("🐍")
            st.subheader("后端语言")
            st.write("Python")
    
    with tech_cards[2]:
        with st.container(border=True):
            st.markdown("🌲")
            st.subheader("机器学习算法")
            st.write("随机森林")
    
    with tech_cards[3]:
        with st.container(border=True):
            st.markdown("📊")
            st.subheader("数据处理")
            st.write("Pandas")
    
    st.divider()
    st.success("🎉 系统已成功运行，欢迎使用学生成绩分析与预测系统！")

# 页面2：专业数据分析
elif page == '专业数据分析':
    # 页面标题和副标题
    st.title("📈 专业数据分析")
    st.subheader("深入洞察各专业学生学业表现")
    st.divider()
    
    # 计算各专业统计数据（用于多个图表）
    major_stats = df.groupby('专业').agg({
        '每周学习时长（小时）': 'mean',
        '期中考试分数': 'mean',
        '期末考试分数': 'mean'
    }).round(2)
    major_stats.columns = ['每周平均学时', '期中考试平均分', '期末考试平均分']
    
    # 1. 各专业学习数据统计 - 现代化表格卡片
    st.subheader("📊 各专业学习数据统计")
    with st.container():
        with st.container(border=True):
            st.dataframe(
                major_stats,
                width='stretch',
                hide_index=False,
                use_container_width=True
            )
    
    st.divider()
    
    # 2. 各专业男女性别比例 - 图表+表格组合
    st.subheader("👥 各专业男女性别比例")
    with st.container():
        # 使用Streamlit原生容器包裹整个模块
        with st.container(border=True):
            # 左右两列布局，使用gap参数增加间距
            gender_cols = st.columns([2, 1], gap="large")
            
            with gender_cols[0]:
                # 计算每个专业的男女人数
                gender_counts = df.groupby(['专业', '性别']).size().unstack(fill_value=0)
                gender_counts = gender_counts[['男', '女']]  # 确保列顺序
                gender_ratio = gender_counts.div(gender_counts.sum(axis=1), axis=0)
                
                # 转换为长格式数据
                gender_ratio_long = gender_ratio.reset_index().melt(id_vars=['专业'], var_name='性别', value_name='比例')
                
                # 使用Plotly创建双列柱状图，优化颜色和样式
                import plotly.express as px
                fig = px.bar(
                    gender_ratio_long,
                    x='专业',
                    y='比例',
                    color='性别',
                    barmode='group',
                    color_discrete_map={'男': '#3b82f6', '女': '#8b5cf6'},
                    category_orders={'性别': ['男', '女']},
                    labels={'比例': '比例', '专业': '专业', '性别': '性别'},
                    height=400,
                    template="plotly_white"
                )
                
                # 优化图表样式
                fig.update_layout(
                    legend_title_text='性别',
                    legend=dict(
                        orientation='h',
                        yanchor='top',
                        y=1.15,
                        xanchor='center',
                        x=0.5
                    ),
                    xaxis_tickangle=0,
                    margin=dict(t=100, b=50),
                    font=dict(family="sans serif"),
                    title_font=dict(size=16, color="#1e293b"),
                    xaxis_title_font=dict(size=14),
                    yaxis_title_font=dict(size=14)
                )
                
                # 显示图表
                st.plotly_chart(fig, use_container_width=True)
            
            with gender_cols[1]:
                # 准备性别比例表格数据
                gender_table = gender_counts.copy()
                gender_table['总人数'] = gender_table['男'] + gender_table['女']
                gender_table['男性比例(%)'] = (gender_table['男'] / gender_table['总人数']).round(2) * 100
                gender_table['女性比例(%)'] = (gender_table['女'] / gender_table['总人数']).round(2) * 100
                gender_table.columns = ['男性人数', '女性人数', '总人数', '男性比例(%)', '女性比例(%)']
                
                # 显示表格
                st.subheader("详细数据")
                st.dataframe(
                    gender_table.round(2), 
                    width='stretch', 
                    use_container_width=True,
                    height=350
                )
    
    st.divider()
    
    # 3. 各专业平均上课出勤率 - 图表+表格组合
    st.subheader("📋 各专业平均上课出勤率")
    with st.container():
        with st.container(border=True):
            attendance_cols = st.columns([2, 1], gap="large")
            
            with attendance_cols[0]:
                # 计算出勤率数据
                attendance_stats = df.groupby('专业')['上课出勤率'].mean().round(4)
                attendance_stats_percent = attendance_stats * 100
                
                # 创建柱状图，优化颜色和样式
                import plotly.express as px
                fig = px.bar(
                    attendance_stats_percent,
                    x=attendance_stats_percent.index,
                    y=attendance_stats_percent.values,
                    labels={'x': '专业', 'y': '平均出勤率(%)'},
                    height=400,
                    color_discrete_sequence=['#10b981'],
                    template="plotly_white"
                )
                
                # 优化图表样式
                fig.update_layout(
                    xaxis_tickangle=0,
                    margin=dict(t=50, b=50),
                    font=dict(family="sans serif"),
                    title_font=dict(size=16, color="#1e293b"),
                    xaxis_title_font=dict(size=14),
                    yaxis_title_font=dict(size=14)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with attendance_cols[1]:
                # 准备出勤率表格数据
                attendance_table = attendance_stats_percent.reset_index()
                attendance_table.columns = ['专业', '平均出勤率(%)']
                
                st.subheader("出勤率数据")
                st.dataframe(
                    attendance_table.round(2), 
                    width='stretch', 
                    use_container_width=True,
                    height=350,
                    hide_index=True
                )
    
    st.divider()
    
    # 4. 各专业期中期末成绩趋势 - 折线图+表格组合
    st.subheader("📈 各专业期中期末成绩趋势")
    with st.container():
        with st.container(border=True):
            comparison_cols = st.columns([2, 1], gap="large")
            
            with comparison_cols[0]:
                # 使用Plotly创建折线图，优化颜色和样式
                import plotly.graph_objects as go
                fig = go.Figure()
                
                # 添加期中考试分数折线（优化颜色）
                fig.add_trace(go.Scatter(
                    x=major_stats.index,
                    y=major_stats['期中考试平均分'],
                    name='期中考试分数',
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=8, color='#3b82f6'),
                    yaxis='y1'
                ))
                
                # 添加期末考试分数折线（优化颜色）
                fig.add_trace(go.Scatter(
                    x=major_stats.index,
                    y=major_stats['期末考试平均分'],
                    name='期末考试分数',
                    mode='lines+markers',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=8, color='#ef4444'),
                    yaxis='y1'
                ))
                
                # 添加每周学习时长折线（优化颜色）
                fig.add_trace(go.Scatter(
                    x=major_stats.index,
                    y=major_stats['每周平均学时'],
                    name='每周学习时长',
                    mode='lines+markers',
                    line=dict(color='#f59e0b', width=3, dash='dash'),
                    marker=dict(size=8, color='#f59e0b'),
                    yaxis='y2'
                ))
                
                # 设置图表布局，优化样式
                fig.update_layout(
                    title='各专业期中期末成绩趋势',
                    xaxis_tickangle=0,
                    xaxis=dict(title='专业'),
                    yaxis=dict(
                        title=dict(text='分数', font=dict(color='#3b82f6')),
                        tickfont=dict(color='#3b82f6')
                    ),
                    yaxis2=dict(
                        title=dict(text='每周学习时长（小时）', font=dict(color='#f59e0b')),
                        tickfont=dict(color='#f59e0b'),
                        anchor='free',
                        overlaying='y',
                        side='right',
                        position=1.0
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='top',
                        y=1.15,
                        xanchor='center',
                        x=0.5
                    ),
                    margin=dict(t=120, r=120, b=50),
                    height=400,
                    template="plotly_white",
                    font=dict(family="sans serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with comparison_cols[1]:
                # 准备成绩对比表格数据
                comparison_table = major_stats[['期中考试平均分', '期末考试平均分', '每周平均学时']].reset_index()
                comparison_table.columns = ['专业', '期中考试分数', '期末考试分数', '每周学习时长']
                comparison_table = comparison_table.round(2)
                
                st.subheader("成绩对比数据")
                st.dataframe(
                    comparison_table, 
                    width='stretch', 
                    use_container_width=True,
                    height=350,
                    hide_index=True
                )
    
    st.divider()
    
    # 5. 大数据管理专业专项分析 - 重点专业深度分析
    st.subheader("🔍 大数据管理专业专项分析")
    with st.container():
        with st.container(border=True):
            # 筛选大数据管理专业数据
            data_science_data = df[df['专业'] == '大数据管理']
            
            # 计算相关指标
            data_science_avg_attendance = data_science_data['上课出勤率'].mean().round(4) * 100
            data_science_avg_final = data_science_data['期末考试分数'].mean().round(2)
            data_science_count = len(data_science_data)
            
            # 使用指标卡片展示，使用Streamlit原生组件
            st.subheader("📊 专业关键指标")
            metric_cols = st.columns(3, gap="medium")
            
            with metric_cols[0]:
                with st.container(border=True):
                    st.subheader("专业人数")
                    st.metric("", data_science_count)
            
            with metric_cols[1]:
                with st.container(border=True):
                    st.subheader("平均出勤率")
                    st.metric("", f"{data_science_avg_attendance:.2f}%")
            
            with metric_cols[2]:
                with st.container(border=True):
                    st.subheader("期末平均分")
                    st.metric("", data_science_avg_final)
            
            # 显示专业详细数据表格
            st.subheader("📋 专业详细数据")
            st.dataframe(
                data_science_data[['性别', '每周学习时长（小时）', '上课出勤率', '期中考试分数', '期末考试分数']],
                width='stretch',
                use_container_width=True,
                height=300,
                hide_index=True
            )
    
    st.divider()
    st.success("✅ 数据分析完成！您可以通过左侧导航查看其他功能模块。")

# 页面3：期末成绩预测
elif page == '期末成绩预测':
    # 页面标题和副标题
    st.title("🤖 期末成绩预测")
    st.subheader("基于机器学习的智能成绩预测")
    st.divider()
    
    # 输入表单区域
    st.subheader("📋 学生信息录入")
    
    with st.container():
        # 使用Streamlit原生容器包裹表单
        with st.container(border=True):
            st.write("请输入学生的相关信息，系统将为您预测期末考试分数。")
            
            with st.form(key='prediction_form', clear_on_submit=False):
                # 表单列布局，使用gap参数增加间距
                form_cols = st.columns(2, gap="large")
                
                with form_cols[0]:
                    # 基本信息
                    st.subheader("🧑‍🎓 基本信息")
                    
                    # 性别选择
                    gender = st.selectbox(
                        "性别", 
                        ['男', '女'], 
                        index=0,
                        help="选择学生的性别"
                    )
                    
                    # 专业选择
                    major = st.selectbox(
                        "专业", 
                        ['工商管理', '人工智能', '财务管理', '电子商务', '大数据管理'], 
                        index=0,
                        help="选择学生的专业"
                    )
                    
                    # 每周学习时长
                    study_hours = st.slider(
                        '每周学习时长（小时）', 
                        min_value=0.0, 
                        max_value=50.0, 
                        step=0.1, 
                        value=15.0,
                        help="学生每周的学习时长"
                    )
                
                with form_cols[1]:
                    # 学习表现
                    st.subheader("📚 学习表现")
                    
                    # 上课出勤率
                    attendance = st.slider(
                        '上课出勤率', 
                        min_value=0.0, 
                        max_value=1.0, 
                        step=0.01, 
                        value=0.8,
                        help="学生的上课出勤率（0.0-1.0）"
                    )
                    
                    # 期中考试分数
                    midterm_score = st.slider(
                        '期中考试分数', 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.1, 
                        value=70.0,
                        help="学生的期中考试分数"
                    )
                    
                    # 作业完成率
                    homework_completion = st.slider(
                        '作业完成率', 
                        min_value=0.0, 
                        max_value=1.0, 
                        step=0.01, 
                        value=0.85,
                        help="学生的作业完成率（0.0-1.0）"
                    )
                
                # 提交按钮，使用Streamlit原生样式
                st.write(" ")  # 空行占位
                submit_button = st.form_submit_button(
                    label='📊 预测成绩',
                    type="primary",
                    use_container_width=True
                )
    
    # 预测结果展示区域
    if submit_button:
        st.divider()
        st.subheader("📊 预测结果")
        
        with st.container():
            # 准备输入数据
            input_data = {
                '性别': 0 if gender == '男' else 1,
                '每周学习时长（小时）': study_hours,
                '上课出勤率': attendance,
                '期中考试分数': midterm_score,
                '作业完成率': homework_completion,
                '专业_工商管理': 1 if major == '工商管理' else 0,
                '专业_人工智能': 1 if major == '人工智能' else 0,
                '专业_财务管理': 1 if major == '财务管理' else 0,
                '专业_电子商务': 1 if major == '电子商务' else 0,
                '专业_大数据管理': 1 if major == '大数据管理' else 0
            }
            
            # 转换为DataFrame
            input_df = pd.DataFrame([input_data])
            
            # 确保特征顺序一致
            input_df = input_df[features]
            
            # 预测期末考试分数
            predicted_score = model.predict(input_df)[0]
            predicted_score_rounded = round(predicted_score, 2)
            
            # 使用Streamlit原生状态消息和布局显示结果
            if predicted_score_rounded >= 60:
                # 及格结果
                with st.container(border=True):
                    st.success(f"🎉 预测成绩及格！")
                    st.write("恭喜！根据模型预测，该学生的期末考试成绩及格。")
                    st.metric("预测期末考试分数", predicted_score_rounded)
                
                result_image = "tongguo.jpg"
            else:
                # 不及格结果
                with st.container(border=True):
                    st.warning(f"⚠️ 预测成绩未及格")
                    st.write("需要继续努力！根据模型预测，该学生的期末考试成绩可能不及格。")
                    st.metric("预测期末考试分数", predicted_score_rounded)
                
                result_image = "guake.jpg"
            
            # 显示相应图片，使用现代化布局
            st.subheader("🖼️ 预测结果可视化")
            
            # 居中显示图片
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(
                        result_image, 
                        width="stretch",
                        caption="预测结果可视化"
                    )
            
            # 添加建议，使用Streamlit原生容器
            st.divider()
            st.subheader("💡 学习建议")
            
            with st.container():
                with st.container(border=True):
                    # 根据预测结果给出不同建议
                    if predicted_score_rounded >= 60:
                        st.write("📚 **保持良好状态**：")
                        st.write("- 继续保持当前的学习节奏和出勤率")
                        st.write("- 可以适当挑战更难的学习内容")
                        st.write("- 帮助其他同学，共同进步")
                    else:
                        st.write("🎯 **提升建议**：")
                        st.write(f"- 增加每周学习时长，建议至少达到 {study_hours + 5:.1f} 小时")
                        st.write("- 提高上课出勤率，争取达到 90% 以上")
                        st.write("- 加强作业完成质量，及时复习巩固")
                        st.write("- 向老师和同学寻求帮助，解决学习难点")
    
    st.divider()
    st.info("ℹ️ 提示：预测结果仅供参考，实际成绩还取决于多种因素。")
