部署流程
1. 安装必要环境
确保已安装以下工具：
Python 3.8 或更高版本
pip 包管理器

Linux (Ubuntu/Debian)
更新系统包：
sudo apt update
sudo apt upgrade
sudo apt install python3 python3-pip


2. 安装依赖
pip install -r requirements.txt

3.api申请
https://api.tu-zi.com/topup

不知道哪里获取Openai或者其他大模型的API，用这个网站！

开发测试的话，5元（RMB），足够了

4.运行需要使用命令行：

streamlit run app.py

后台运行：
nohup streamlit run app.py > streamlit.log 2>&1 &

访问http://xxx:8501/
正常可开始使用





