
部署流程

1.安装必要环境
确保已安装以下工具：
Python 3.8 或更高版本
pip 包管理器

Linux (Ubuntu/Debian)
更新系统包：
sudo apt update
sudo apt upgrade
sudo apt install python3 python3-pip

2.安装依赖
pip install -r requirements.txt

3.api申请
https://api.tu-zi.com/topup

不知道哪里获取Openai或者其他大模型的API，用这个网站！

开发测试的话，5元（RMB），足够了

在代码中仅修改OPENAI_API_KEY = "xxxx" #更新为自己的APIKey这里即可开始运行使用。默认使用grok-3-reasoner模型，需要其他请在代码中修改替换

4.运行需要使用命令行：

app.py包含4个AI分析系统包含：
市场行为分析
持仓分析
多周期分析
资金流向分析（原始版本）
streamlit run app.py
文件说明，按需要下载文件运行程序即可
feilv.py 仅是费率监控程序，需要单独再运行
zijinliu.py 资金流向分析（T姐二改版本）


后台运行（运行那个程序就替换app.py这个文件即可）：
app.py

nohup streamlit run app.py > streamlit.log 2>&1 &



zijinliu.py 资金流向分析（T姐二改版本）
nohup streamlit run zijliu.py.py > zijliu.py.log 2>&1 &

停止后台进程
如果需要停止服务，可以找到进程 ID 并杀掉。
查找进程

ps -ef | grep -E "streamlit|python"

输出示例：

root  12345  ...  streamlit run feilv.py -- --mode frontend
root  12346  ...  python feilv.py --mode backend

杀死进程
bash

kill -9 12345  # 停止前端
kill -9 12346  # 停止后端





访问http://xxx:8501/
正常可开始使用

开箱即用下载app.py即可，其他文件都是T姐的源码，如有需要可以二次开发使用，感谢原作者T姐Theclues，配合炒币工具箱效果更佳，https://t.me/SuperClues_bot



