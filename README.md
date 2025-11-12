# AI Agent Powered Data Analysis Chatbot

## Installation

This project was built on Ubuntu 22.04.5 LTS and has been successfully tested under Python 3.10 and Python 3.11.

* Download source code:

```shell
git clone https://github.com/ferry2174/data_analysis_with_agent.git
```

* It is recommended to create a Python virtual environment with version 3.11 and then install it from the source package.

```shell
cd data_analysis_with_agent
pip install -e .
```

* Startup gradio program

```shell
start_dawa_app.sh
```

When you see `"Running on local URL: http://0.0.0.0:7860"`, it means the application has started successfully. Open your web browser and enter <http://127.0.0.1:7860> to access the application.

* Launch a web service built on FastAPI to display additional graph components.

```shell
start_dawa_app.sh dev
```

Open your web browser and visit `http://127.0.0.1:8090/data_analysis_with_agent`. If the page displays `"Welcome to Python restful API Project Template!"`, the REST service has started successfully.

## Try out

* The demo program currently only supports one dataset: data_analysis_with_agent/src/data_analysis_with_agent/assets/example_data_gang_profits.zip

* Click the "上传ZIP文件" button on the page, select the file, and you can see basic information about this dataset.Click the "Upload File" button on the page and select the file. After uploading, you can see the data processing process on the interface, and then you can see basic information about this dataset.

* Try using the following prompts:
  1. View clients list
  2. Check Luo Gang's data
  3. View data for transactions exceeding 50,000 yuan in March 2021
  4. Perform transaction network analysis
  You will see the analyzed data in the upper right corner and a data visualization in the lower right corner.
* Currently supported analysis methods include:
  1. Time Pattern Analysis: Analyzing the temporal regularity of transactions
  2. Amount Network Analysis: Analyzing the network relationships of transaction amounts
  3. Balance Anomaly Analysis: Detecting abnormal balance change patterns
  4. Transaction Cycle Analysis: Identifying possible cyclical transaction patterns
  5. Comprehensive Group Profit Analysis: Combining time patterns, transaction networks, balance anomalies, and transaction cycles for comprehensive analysis
* Since this is a demo project, the analysis capabilities are not yet fully developed. I will expand upon it further if you have any specific needs.

## Others

Using DeepSeek as the driving model for both the AI ​​agent and AI chatbot, I'm using my personal API key, which, of course, requires payment. You can use it with confidence as long as there's a balance in my personal account. If my account runs out of balance, you can modify the configuration to use your own DeepSeek API key:`data_analysis_with_agent/src/data_analysis_with_agent/config/config_dev.yaml` -> deepseek.api_key.