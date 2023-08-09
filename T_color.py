import streamlit as st

# 设置页面标题
st.title("色块区隔版面示例")
Specimen_Name = st.sidebar.text_input("Specimen Name")

# 创建不同颜色的色块
st.markdown('<div style="background-color: #FFD700; padding: 10px;">'
            '<h2 style="color: black;">这是黄色色块</h2></div>', unsafe_allow_html=True)

st.markdown('<div style="background-color: #87CEEB; padding: 10px;">'
            '<h2 style="color: black;">这是天蓝色色块</h2></div>', unsafe_allow_html=True)

# 在色块中添加内容
st.markdown('<div style="background-color: #98FB98; padding: 10px;">'
            '<h2 style="color: black;">这是淡绿色色块</h2>'
            '<p>在这里可以添加更多内容。</p></div>', unsafe_allow_html=True)
