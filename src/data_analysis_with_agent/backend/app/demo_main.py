import argparse
import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import font_manager


parser = argparse.ArgumentParser(description='EI AI Prototype Demo')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='server startup ip')
parser.add_argument('--knowledge_graph_server_ip', type=str, default='http://127.0.0.1:8090/data_analysis_with_agent/graph/', help='knowledge graph server ip')

args = parser.parse_args(sys.argv[1:])

print(f"server_name: {args.server_name}, knowledge_graph_server_ip: {args.knowledge_graph_server_ip}")

graph_data_dir = os.path.join(os.path.expanduser("~"), "Program", "data_analysis_with_agent", "graph")

"""
Safely dump data to JSON, handling common non-serializable objects
"""
def safe_json_dump(data):
    """
    Safely dump data to JSON, handling common non-serializable objects
    """
    def _convert_value(val):
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        elif isinstance(val, date):
            return val.isoformat()
        elif isinstance(val, pd.Series):
            return val.tolist()
        elif isinstance(val, pd.DataFrame):
            return val.to_dict(orient='records')
        elif hasattr(val, 'dtype'):
            try:
                return val.item()
            except Exception:
                return str(val)
        elif isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        return val

    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_serializable(item) for item in obj]
        else:
            return _convert_value(obj)

    return _make_serializable(data)
# 只在直接运行时解析参数，在 Notebook 中使用默认值
#if 'ipykernel' not in sys.modules:
#    args = parser.parse_args()
#else:
#    args = parser.parse_args([])

### 时间模式分析相关函数
def detect_time_anomalies(df):
    """检测异常时间模式"""
    # 确保时间列是datetime类型
    df['交易时间'] = pd.to_datetime(df['交易时间'])

    # 提取小时和星期几
    df['交易小时'] = df['交易时间'].dt.hour
    df['交易星期'] = df['交易时间'].dt.strftime('%Y-%W')  # 修改为yyyy-ww格式

    # 计算每小时交易量
    hourly_counts = df['交易小时'].value_counts().sort_index()

    # 识别异常时间段（晚上10点到凌晨4点）
    night_hours = list(range(22, 24)) + list(range(0, 5))
    night_trans = df[df['交易小时'].isin(night_hours)]
    night_ratio = round(len(night_trans) / len(df), 4)
    night_ratio = 0 if night_ratio < 0.0001 else night_ratio  # 处理极小值

    # 计算全局统计数据
    total_accounts = df['查询账号'].nunique()
    total_transactions = len(df)
    total_amount = df['交易金额'].sum()

    # 将时间转换为5分钟窗口
    df['时间窗口'] = df['交易时间'].dt.floor('5min')

    # 统计每个窗口内的同步交易
    sync_trans = df.groupby('时间窗口').agg(
        账户数=('查询账号', 'nunique'),
        交易笔数=('交易金额', 'count'),
        总金额=('交易金额', 'sum')
    ).sort_values('总金额', ascending=False).head(10)

    # 计算百分比并添加到结果
    sync_trans['账户数占比'] = (sync_trans['账户数'] / total_accounts * 100).round(2)
    sync_trans['交易笔数占比'] = (sync_trans['交易笔数'] / total_transactions * 100).round(2)
    sync_trans['总金额占比'] = (sync_trans['总金额'] / total_amount * 100).round(2)

    # 转换时间戳为字符串
    top_sync_dict = sync_trans.reset_index()
    top_sync_dict['时间窗口'] = top_sync_dict['时间窗口'].astype(str)
    top_sync_dict = top_sync_dict.to_dict(orient='records')

    return {
        "hourly_distribution": hourly_counts.to_dict(),
        "night_ratio": night_ratio,
        "top_sync_times": top_sync_dict,
        "weekday_pattern": df['交易星期'].value_counts().sort_index().to_dict(),
        "global_stats": {
            "total_accounts": total_accounts,
            "total_transactions": total_transactions,
            "total_amount": total_amount
        }
    }

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def generate_time_visualization(results):
    """生成时间分析可视化"""
    # 加载本地字体文件
    font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'STKAITI.TTF')
    font_prop = FontProperties(fname=font_path)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 小时分布图
    hours = list(range(24))
    counts_ax1 = [results['hourly_distribution'].get(str(h), 0) for h in hours]  # 注意键是字符串
    ax1.bar(hours, counts_ax1)
    ax1.set_title('每小时交易量分布', fontproperties=font_prop)
    ax1.set_xlabel('小时', fontproperties=font_prop)
    ax1.set_ylabel('交易次数', fontproperties=font_prop)

    # 设置x轴刻度标签的字体
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 修改后的周分布图 - 展示交易次数排名前10的周
    # 从weekday_pattern中提取数据并排序
    week_data = results['weekday_pattern']
    # 按交易量排序并取前10
    sorted_weeks = sorted(week_data.items(), key=lambda x: x[1], reverse=True)[:10]
    weeks = [week[0] for week in sorted_weeks]
    counts_ax2 = [week[1] for week in sorted_weeks]

    ax2.bar(weeks, counts_ax2)
    ax2.set_title('交易量排名前10的周分布', fontproperties=font_prop)
    ax2.set_xlabel('周(年-周数)', fontproperties=font_prop)
    ax2.set_ylabel('交易次数', fontproperties=font_prop)
    ax2.tick_params(axis='x', rotation=45)  # 旋转x轴标签以便更好显示

    # 设置x轴刻度标签的字体
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_prop)

    try:
        plt.tight_layout()
    except Exception as e:
        print(f"调整布局时出错: {e}")
    return fig

def format_time_results(results):
    """格式化时间分析结果"""
    report = [
        "## 时间模式分析结果",
        f"- 夜间交易占比: {results['night_ratio']:.1%}",
        f"- 交易高峰时段: {max(results['hourly_distribution'].items(), key=lambda x: x[1])[0]}时",
    ]
    return "\n".join(report)

### 交易网络分析相关函数
def build_amount_networks(df):
    """构建交易金额网络"""
    print("in build_amount_networks")
    G = nx.DiGraph()
    # 添加所有节点（账户）
    #all_accounts = set(df['查询账号']).union(set(df['交易对方账号']))
    df[['客户名称', '交易对方名称']] = df[['客户名称', '交易对方名称']].fillna('未知姓名')
    all_accounts = set(df['客户名称']).union(set(df['交易对方名称']))

    for acc in all_accounts:
        G.add_node(acc)

    # 添加边和交易属性
    for _, row in df.iterrows():
        source = row['客户名称']
        target = row['交易对方名称']
        amount = row['交易金额']

        if source and target:
            if G.has_edge(source, target):
                G[source][target]['amount'] += amount
                G[source][target]['count'] += 1
            else:
                G.add_edge(source, target, amount=amount, count=1)

    # 删除金额小于10万的边
    G_undirected = G.to_undirected()
    edges_to_remove = [(u, v) for u, v, d in G_undirected.edges(data=True) if d.get('amount', 0) < 200000]

    G_undirected.remove_edges_from(edges_to_remove)
    G.remove_edges_from(edges_to_remove)
    G_undirected.remove_nodes_from(list(nx.isolates(G_undirected)))
    G.remove_nodes_from(list(nx.isolates(G)))

    centrality_dict = {}
    communities_dict = {}
    # 计算网络指标
    if len(G) > 0:
        centrality = nx.degree_centrality(G_undirected)
        try:
            # 尝试计算社区
            import community as community_louvain  # pip install python-louvain
            communities = community_louvain.best_partition(G_undirected, weight='amount', resolution=0.1)
            num_communities = len(set(communities.values()))
            modularity = community_louvain.modularity(communities, G_undirected)
            print(f"成功检测到 {num_communities} 个社区")
            print(f"模块度：{modularity:.3f}")
            from collections import Counter
            community_sizes = Counter(communities.values())
            print("社区大小分布:", community_sizes.most_common())
            print(f"centrality: {centrality}")
            # 1. 按度中心性排序（前10节点）
            top10_centrality = sorted(
                centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            print("\n=== 度中心性 Top 10 ===")
            for node, cent in top10_centrality:
                print(f"节点: {node}, 中心性: {cent:.4f}")

            centrality_dict = {name: value for name, value in top10_centrality}

            # 2. 按社区大小排序（前10社区）
            community_members = {}
            for node, comm_id in communities.items():
                if comm_id not in community_members:
                    community_members[comm_id] = []
                community_members[comm_id].append(node)

            top10_communities = sorted(
                community_members.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
            print("\n=== 最大社区 Top 10 ===")
            for comm_id, members in top10_communities:
                print(f"社区ID: {comm_id}, 节点数: {len(members)}, 示例节点: {members[:5]}...")

            communities_dict = {name: value for name, value in top10_communities}
        except Exception as e:  # 捕获所有可能的异常
            print(f"社区检测失败，错误信息: {e}")
            communities = []  # 返回空列表或其他默认值
    else:
        centrality = {}
        communities = []

    # 识别大额交易
    large_trans = df[df['交易金额'] > df['交易金额'].quantile(0.98)]
#    large_trans['时间窗口'] = large_trans['交易时间'].dt.floor('5min')
    large_trans['交易时间'] = large_trans['交易时间'].astype(str)

    return {
        "graph": G,
        "centrality": centrality_dict,
        "communities": communities_dict,
        "large_transactions": large_trans.to_dict('records'),
        "network_stats": {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "density": nx.density(G)
        }
    }

def create_network_graph(results):
    """创建交易网络可视化"""
    G = results['graph']
    # 简化网络：只显示重要节点
    centrality = results['centrality']
    top_nodes = sorted(centrality.items(), key=lambda x: -x[1])[:20]
    top_nodes = [n[0] for n in top_nodes]
    subgraph = G.subgraph(top_nodes)

    font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'STKAITI.TTF')
    font_manager.fontManager.addfont(font_path)
    zh_font = FontProperties(fname=font_path)
    font_name = zh_font.get_name()
    plt.rcParams['font.family'] = font_name
    # 绘制网络图
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph)

    # 节点大小反映中心性
    node_sizes = [centrality[n]*5000 for n in subgraph.nodes()]

    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='skyblue')
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_family=font_name)

    plt.title('交易网络拓扑结构（关键节点）', fontproperties=zh_font)
    plt.axis('off')

    # 保存图片
#    img_path = f"temp_data/network_{uuid.uuid4()}.png"
    img_path = os.path.join(graph_data_dir, f"network_{uuid.uuid4()}.png")
    plt.savefig(img_path)
    plt.close()

    # 返回路径和生成状态
    return img_path, True, img_path, True

def format_network_results(results):
    """格式化网络分析结果"""
    print("in format_network_results")
    report = [
        "## 交易网络分析结果",
        f"- 网络规模: {results['network_stats']['nodes']}个节点, {results['network_stats']['edges']}条边",
        f"- 网络密度: {results['network_stats']['density']:.3f}",
        f"- 检测到{len(results['communities'])}个潜在团伙",
        "- 关键账户:"
    ]

    # 添加前5个关键账户
    top_accounts = sorted(results['centrality'].items(), key=lambda x: -x[1])[:5]
    for acc, score in top_accounts:
        report.append(f"  - {acc} (中心性: {score:.2f})")

    return "\n".join(report)

### 余额异常分析相关函数
def detect_balance_anomalies(df):
    """检测余额异常模式"""
    df = df.sort_values(['查询账号', '交易时间'])
    anomalies = []
    
    # 检测快速清零模式
    for acc, group in df.groupby('查询账号'):
        if len(group) < 2:
            continue
            
        for i in range(1, len(group)):
            prev = group.iloc[i-1]
            curr = group.iloc[i]
            
            if prev['借贷标志'] == '入' and curr['借贷标志'] == '出':
                time_diff = (curr['交易时间'] - prev['交易时间']).total_seconds() / 60
                amount_ratio = curr['交易金额'] / prev['交易金额']
                
                if time_diff < 120 and amount_ratio > 0.5:  # 2小时内转出80%以上
                    anomalies.append({
                        '账号': acc,
                        '类型': '快速清零',
                        '时间差(分钟)': time_diff,
                        '入账金额': prev['交易金额'],
                        '转出金额': curr['交易金额'],
                        '转出比例': amount_ratio
                    })
    
    # 检测阶梯式变动
    balance_stats = df.groupby('查询账号')['交易余额'].agg(['mean', 'std'])
    balance_stats['cv'] = balance_stats['std'] / balance_stats['mean']
    stepped_accounts = balance_stats[balance_stats['cv'] < 1.15].index.tolist()
    
    return {
        "rapid_clear": anomalies,
        "stepped_accounts": stepped_accounts,
        "balance_cv": balance_stats['cv'].to_dict()  # 修改为返回原始cv数据
    }

def generate_balance_visualization(results):
    """生成余额异常可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'STKAITI.TTF')
    font_prop = FontProperties(fname=font_path)
    # 快速清零统计
    if results['rapid_clear']:
        clear_times = [x['时间差(分钟)'] for x in results['rapid_clear']]
        ax1.hist(clear_times, bins=20)
        ax1.set_title('快速清零时间分布', fontproperties=font_prop)
        ax1.set_xlabel('分钟', fontproperties=font_prop)
        ax1.set_ylabel('次数', fontproperties=font_prop)
    
    # 余额波动统计
    if 'balance_cv' in results:
        cv_values = list(results['balance_cv'].values())  # 获取所有cv值
        ax2.hist(cv_values, bins=20)
        ax2.set_title('余额变异系数分布', fontproperties=font_prop)

    plt.tight_layout()
    return fig

def format_balance_results(results):
    """格式化余额分析结果（适配最新数据结构）"""
    report = [
        "## 余额异常分析结果",
        f"- 检测到 {len(results['rapid_clear'])} 次快速清零操作",
        f"- 发现 {len(results['stepped_accounts'])} 个阶梯式变动账户"
    ]

    # 添加CV统计信息（如果存在）
    if 'balance_cv' in results and results['balance_cv']:
        cv_values = list(results['balance_cv'].values())
        mean_cv = sum(cv_values) / len(cv_values) if cv_values else 0
        max_cv = max(cv_values) if cv_values else 0

        report.extend([
            "- 余额变异系数统计:",
            f"  - 平均CV: {mean_cv:.4f}",
            f"  - 最大CV: {max_cv:.4f}"
        ])

    return "\n".join(report)

### 交易循环分析相关函数
def detect_transaction_cycles(df, max_cycle_length=5):
    """检测资金循环交易"""
    G = nx.DiGraph()
    
    # 构建交易图（只包含大额交易）
    large_trans = df[df['交易金额'] > df['交易金额'].quantile(0.9)]
    for _, row in large_trans.iterrows():
        G.add_edge(row['查询账号'], row['交易对方账号'], amount=row['交易金额'])
    
    # 检测简单循环
    cycles = []
    for cycle in nx.simple_cycles(G):
        if 3 <= len(cycle) <= max_cycle_length:
            # 计算循环中的最小交易金额
            min_amount = min(
                G[cycle[i]][cycle[(i+1)%len(cycle)]]['amount'] 
                for i in range(len(cycle))
            )
            cycles.append({
                'path': cycle,
                'length': len(cycle),
                'min_amount': min_amount
            })
    
    # 按金额排序
    return sorted(cycles, key=lambda x: -x['min_amount'])[:10]  # 返回前10个

def generate_cycle_visualization(results):
    font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'STKAITI.TTF')
    font_prop = FontProperties(fname=font_path)

    """生成交易循环可视化"""
    if not results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, '未检测到显著资金循环', ha='center', fontproperties=font_prop)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 显示每个循环的金额和长度
    cycles = sorted(results, key=lambda x: -x['min_amount'])
    y_pos = range(len(cycles))
    amounts = [c['min_amount'] for c in cycles]
    lengths = [c['length'] for c in cycles]
    
    bars = ax.barh(y_pos, amounts, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"循环-{i+1}" for i in y_pos], fontproperties=font_prop)
    ax.set_xlabel('最小交易金额', fontproperties=font_prop)
    ax.set_title('检测到的资金循环', fontproperties=font_prop)
    
    # 在条上显示长度
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, i, f'长度: {lengths[i]}', ha='left', va='center', fontproperties=font_prop)
    
    plt.tight_layout()
    return fig

def format_cycle_results(results):
    """格式化循环分析结果"""
    if not results:
        return "## 交易循环分析\n- 未检测到显著资金循环模式"
    
    report = [
        "## 交易循环分析结果",
        f"- 检测到{len(results)}个资金循环模式",
        "- 主要循环:"
    ]
    
    for i, cycle in enumerate(results[:3], 1):
        report.append(
            f"  {i}. 路径: {' → '.join(cycle['path'])} "
            f"(金额: {cycle['min_amount']:.2f}, 长度: {cycle['length']})"
        )
    
    return "\n".join(report)


### 综合分析与风险评估函数
def calculate_risk_score(time_results, network_results, balance_results, cycle_results):
    """计算综合风险分数"""
    print(f"in calculate_risk_score 1")
    risk_factors = {
        'time_risk': min(time_results['night_ratio'] * 10, 1.0),
        'network_risk': len(network_results['communities']) / 10,
        'balance_risk': len(balance_results['rapid_clear']) / 20,
        'cycle_risk': len(cycle_results) / 5
    }

    print(f"risk_factors: {risk_factors}")
    
    # 加权计算
    weights = {
        'time_risk': 0.2,
        'network_risk': 0.3,
        'balance_risk': 0.3,
        'cycle_risk': 0.2
    }
    
    total_risk = sum(risk_factors[k] * weights[k] for k in risk_factors)

    return {
        'score': min(total_risk, 1.0),
        'factors': risk_factors,
        'weights': weights
    }

def generate_comprehensive_report(results):
    """生成综合分析报告"""
    print("in generate_comprehensive_report 111111111")
    report = [
        "# 团伙获利综合分析报告",
        f"## 综合风险分数: {results['risk_score']['score']:.2f}/1.0",
        "### 风险因素分析:",
        f"- 时间风险: {results['risk_score']['factors']['time_risk']:.2f}",
        f"- 网络风险: {results['risk_score']['factors']['network_risk']:.2f}",
        f"- 余额风险: {results['risk_score']['factors']['balance_risk']:.2f}",
        f"- 循环风险: {results['risk_score']['factors']['cycle_risk']:.2f}",
        "\n## 详细分析结果"
    ]
    # 添加各模块摘要
    report.append(format_time_results(results['time_analysis']))
    report.append(format_network_results(results['network_analysis']))
    report.append(format_balance_results(results['balance_analysis']))
    report.append(format_cycle_results(results['cycle_analysis']))

    # 添加建议
    risk_level = "高风险" if results['risk_score']['score'] > 0.7 else \
               "中风险" if results['risk_score']['score'] > 0.4 else "低风险"

    report.append(f"\n## 建议\n- 当前检测结果为{risk_level}等级")
    if risk_level == "高风险":
        report.append("- 建议立即进行人工核查")
        report.append("- 重点关注网络分析中的关键账户")
    elif risk_level == "中风险":
        report.append("- 建议抽样核查可疑交易")

    return "\n".join(report)

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import uuid
import json
import math
from copy import deepcopy
from threading import Semaphore
from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk, ToolMessage
from data_analysis_with_agent.backend.app.knowledgebase import compress_data

from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatOllama
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "sk-233a5931660b4d06b0725e6ca7836495"

# 初始化DeepSeek模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    origin_datapath: str
    data_file_path: dict
    current_data_file_path: str
    tips: str

graph_builder = StateGraph(State)

def get_transaction_by_name(name: str, state: State):
    """
    根据客户名称获取交易信息。
    Param:
        - name: 客户名称
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）
        chart_type: str: 不支持图表输出
    """
    print("start tool: get_transaction_by_name")
    input_path = state["origin_datapath"]
    if not input_path:
        raise ValueError("分析必须依赖于一个文件作为上下文，请先上传一个文件！")
    df = pd.read_csv(input_path)
    if name:
        df = df[df["客户名称"] == name]

#    data_path = f"temp_data/{str(uuid.uuid4())}.csv"
    data_path = os.path.join(graph_data_dir, f"{str(uuid.uuid4())}.csv")
    state["current_data_file_path"] = data_path
    key = f"transaction_by_name_{name}"
    state["data_file_path"][key] = data_path

    df.to_csv(data_path, index=False, encoding="utf-8-sig")
    return {"data_path": data_path, "data_type": "csv", "table_type": "dataframe", "chart_type": "", "tips": ""}

from collections import Counter
def get_customer_list(state: State, original_path: str):
    """
    获取客户列表，以及一些统计信息。
    Param:
        - state: 当前状态
        - original_path: 原始数据路径
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）
        chart_type: str: 不支持图表输出
    """
    print("start tool: get_customer_list")
    if not original_path:
        raise ValueError("分析必须依赖于一个文件作为上下文，请先上传一个文件！")
    df = pd.read_csv(original_path)

    # 1. 针对客户名称去重
    unique_customers = df['客户名称'].unique()
    results = []
    for customer in unique_customers:
        customer_data = df[df['客户名称'] == customer]
        # 2. 对应的证件号码 (取第一个，因为客户名称已去重，证件号码应该相同)
        if customer_data.empty:
            continue
        id_number = customer_data['客户证件号码'].iloc[0]
        # 3. 常用查询账号列表
        query_accounts = customer_data['查询账号'].dropna().unique()
        query_accounts_str = ",".join(str(acc) for acc in query_accounts) if len(query_accounts) > 0 else ""
        # 4. 常用查询卡号列表
        query_cards = customer_data['查询卡号'].dropna().unique()
        query_cards_str = ",".join(str(card) for card in query_cards) if len(query_cards) > 0 else ""
        # 5. 交易对方数量
        counterparties = customer_data['交易对方名称'].dropna().nunique()
        # 6. 最大余额、最小余额
        balances = customer_data['交易余额'].dropna()
        max_balance = balances.max() if not balances.empty else None
        min_balance = balances.min() if not balances.empty else None
        # 7. 常用IP地址列表 (取出现频率最高的3个)
        ip_list = customer_data['本方IP地址'].dropna().tolist()
        common_ips = [ip for ip, count in Counter(ip_list).most_common(3)] if ip_list else []
        common_ips_str = ",".join(common_ips)
        # 8. 常用MAC地址列表 (取出现频率最高的3个)
        mac_list = customer_data['本方MAC地址'].dropna().tolist()
        common_macs = [mac for mac, count in Counter(mac_list).most_common(3)] if mac_list else []
        common_macs_str = ",".join(common_macs)
        # 9. 常用交易渠道 (取出现频率最高的3个)
        channel_list = customer_data['交易渠道'].dropna().tolist()
        common_channels = [channel for channel, count in Counter(channel_list).most_common(3)] if channel_list else []
        common_channels_str = ",".join(common_channels)
        results.append({
            '客户名称': customer,
            '客户证件号码': id_number,
            '常用查询账号列表': query_accounts_str,
            '常用查询卡号列表': query_cards_str,
            '交易对方数量': counterparties,
            '最大余额': max_balance,
            '最小余额': min_balance,
            '常用IP地址列表': common_ips_str,
            '常用MAC地址列表': common_macs_str,
            '常用交易渠道': common_channels_str
        })

    df = pd.DataFrame(results)

    uuid_str = str(uuid.uuid4())
#    data_path = f"temp_data/{uuid_str}.csv"
    data_path = os.path.join(graph_data_dir, f"{uuid_str}.csv")
    state["current_data_file_path"] = data_path
    key = f"transaction_by_name_{uuid_str}"
    state["data_file_path"][key] = data_path

    df.to_csv(data_path, index=False, encoding="utf-8-sig")
    requirement = "没有进一步需求。"
    return {"data_path": data_path, "data_type": "csv", "table_type": "dataframe", "chart_type": "", "tips": "", "requirement": requirement}

def get_knowledgebase_data(state: State, original_path: str):
    """
    获取知识图谱数据。
    数据获取后可用于“基于关键信息提取的团伙获利分析”。
    Param:
        - state: 当前状态
        - original_path: 原始数据路径
    return: 返回值是一个json结构
        data_path: str: 提取后的知识图谱数据存储路径
        data_type: str: 数据类型，知识图谱数据用json格式存储
        table_type: str: 表格类型，知识图谱数据用json表格展示
        chart_type: str: 可以支持输出network_graph（网络图谱）
    """
    print("start tool: get_knowledgebase_data")
    # 生成唯一的输出文件名
    input_path = original_path
    if not input_path:
        raise ValueError("分析必须依赖于一个文件作为上下文，请先上传一个文件！")
#    output_json = f"temp_data/knowledge_{str(uuid.uuid4())}.json"
    output_json = os.path.join(graph_data_dir, f"knowledge_{str(uuid.uuid4())}.json")
    tips = ""
    requirement = ""
    try:
        compress_data(input_path, output_json)
        state["current_data_file_path"] = output_json
    except Exception as e:
        print(f"知识提取失败: {str(e)}")
    return { 
        "data_path": output_json, 
        "data_type": "json", 
        "table_type": "json", 
        "chart_type": "network_graph", 
        "tips": tips, 
        "requirement": requirement 
    }

def analysis_gang_profits_with_data_fragment(state: State, original_path: str):
    """
    用数据分片的方法进行团伙获利：数据分片交由大模型分片整理关键信息，再汇总分析的方式。
    Param:
        - state: 当前状态
        - original_path: 原始数据路径
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）
        chart_type: str: 不支持图表输出
    """
    print("start tool: analysis_gang_profits_with_data_fragment")
    tips = "告诉用户你选择了数据分片的方法做的分析，还可以问一下用户是否选择关键信息提取的方法做。"
    prompt_template = """下面是{total}个文件中的其中一个，里面包含了可供分析团伙获利行为的部分数据，
提取其中的关键信息，后面会把这些关键信息进行整合，整合后的信息会再次提交给大模型，
整体分析团伙获利行为的可能性。"""

    # 读取原始CSV文件获取总行数
    if not original_path:
        raise ValueError("分析必须依赖于一个文件作为上下文，请先上传一个文件！")
    df = pd.read_csv(original_path)
    chunk_size = 300
    total_rows = len(df)
    total_chunks = math.ceil(total_rows / chunk_size)

    # 分割文件为多个临时文件
#    temp_path = f"temp_data/{uuid.uuid4()}"
    temp_path = os.path.join(graph_data_dir, f"{uuid.uuid4()}")
    os.makedirs(temp_path, exist_ok=True)
    temp_files = []
    if total_chunks > 2:
        total_chunks = 2
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_df = df.iloc[start_idx:end_idx]
        temp_file = f"{temp_path}/temp_chunk_{i}.csv"
        chunk_df.to_csv(temp_file, index=False)
        temp_files.append(temp_file)

    # 定义分析单个文件的函数
    def analyze_single_chunk(file_path, chunk_num, total):
        print(f"开始分析第 {chunk_num+1} 个文件。")
        try:
            with open(file_path, 'r') as f:
                data = f.read()

            prompt = prompt_template.format(total=total) + f"\n\n这是第{chunk_num+1}个文件的数据:\n{data}"
            response = llm.invoke(prompt)

            print(f"分析第 {chunk_num+1} 个文件已完成。")

            # 返回结果和原始文件路径以便清理
            return {
                'chunk_num': chunk_num,
                'file_path': file_path,
                'analysis': response
            }
        except Exception as e:
            print(f"分析第 {chunk_num+1} 个文件时出错: {str(e)}")
            return {
                'chunk_num': chunk_num,
                'file_path': file_path,
                'analysis': f"分析出错: {str(e)}"
            }

    max_concurrent = 3
    # 使用信号量控制并发量
    semaphore = Semaphore(max_concurrent)
    def analyze_with_semaphore(file_path, chunk_num, total):
        with semaphore:
            return analyze_single_chunk(file_path, chunk_num, total)

    # 使用多线程并行分析
    all_results = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # 提交所有任务
        futures = {executor.submit(analyze_with_semaphore, temp_file, i, total_chunks): i 
                  for i, temp_file in enumerate(temp_files)}
        # 按照完成顺序处理结果
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                over = False
                if len(all_results) == total_chunks:
                    over = True
                print(f"已完成分块 {result['chunk_num']+1}/{total_chunks}")
            except Exception as e:
                print(f"任务执行出错: {str(e)}")

    # 清理临时文件
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass

    # 按原始顺序排序结果
    all_results.sort(key=lambda x: x['chunk_num'])
    combined_analysis = "\n".join([res['analysis'].content for res in all_results])
    requirement = f"""
    下面是整合后的团伙获利关键信息，由 {total_chunks} 个文件片段通过大模型分析得来，根据以下信息用大模型进行分析是否存在团伙获利行为，并反馈分析结果。
    <analysis_data>
    {combined_analysis}
    </analysis_data>
    """

#    output_json = f"temp_data/knowledge_{str(uuid.uuid4())}.json"
    output_json = os.path.join(graph_data_dir, f"knowledge_{str(uuid.uuid4())}.json")
    try:
        compress_data(original_path, output_json)
        state["current_data_file_path"] = output_json
    except Exception as e:
        print(f"知识提取失败: {str(e)}")
    return {"data_path": output_json, "data_type": "json", "table_type": "json", "chart_type": "network_graph", "tips": tips, "requirement": requirement}

def analysis_gang_profits_with_key_info(state: State, original_path: str):
    """
    用关键信息提取的方法进行团伙获利：数据分片交由大模型分片整理关键信息，再汇总分析的方式。
    Param:
        - state: 当前状态
        - original_path: 原始数据路径
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）
        chart_type: str: 不支持图表输出
    """
    print("start tool: analysis_gang_profits_with_key_info")
    # 生成唯一的输出文件名
    input_path = original_path
    if not input_path:
        raise ValueError("分析必须依赖于一个文件作为上下文，请先上传一个文件！")
#    output_json = f"temp_data/knowledge_{str(uuid.uuid4())}.json"
    output_json = os.path.join(graph_data_dir, f"knowledge_{str(uuid.uuid4())}.json")
    requirement = ""
    try:
        compress_data(input_path, output_json)
        state["current_data_file_path"] = output_json
    except Exception as e:
        print(f"知识提取失败: {str(e)}")
    
    with open(output_json, 'r') as f:
        key_info = json.load(f)

    tips = "告诉用户你选择了关键信息提取的方法做的分析，还可以问一下用户是否选择数据分片的方法做。"
    requirement = f"""
        下面是团伙获利分析关键信息，从原始资金交易数据中提取，根据以下信息用大模型进行分析是否存在团伙获利行为，并反馈分析结果。
        <analysis_data>
        {key_info}
        </analysis_data>
    """
    return { 
        "data_path": output_json, 
        "data_type": "json", 
        "table_type": "json", 
        "chart_type": "network_graph", 
        "tips": tips, 
        "requirement": requirement 
    }

# ================ 增强分析方法工具集 ================

def analyze_time_patterns(state: State, original_path: str):
    """
    时间模式分析工具。
    此工具是团伙获利综合分析的一部分，也可以单独使用。
    Param:
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）和 json
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）和 json
        chart_type: str: 目前支持image（图片）
        chart_path: str: 如果chart_type为image，则chart_path为生成的图片路径
        tips: str: 分析提示
        requirement: str: 格式化后的分析结果，可直接用于展示，或供大模型做进一步分析
    """
    print("start tool: analyze_time_patterns")
    df = get_current_data(state)
    results = detect_time_anomalies(df)

#    output_path = f"temp_data/time_analysis_{uuid.uuid4()}.json"
    output_path = os.path.join(graph_data_dir, f"time_analysis_{uuid.uuid4()}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f)

    visualization = generate_time_visualization(results)
#    viz_path = f"temp_data/time_viz_{uuid.uuid4()}.png"
    viz_path = os.path.join(graph_data_dir, f"time_viz_{uuid.uuid4()}.png")
    visualization.savefig(viz_path)

    return {
        "data_path": output_path,
        "data_type": "json",
        "table_type": "json",
        "chart_type": "image",
        "chart_path": viz_path,
        "tips": "时间模式分析已完成",
        "requirement": format_time_results(results)
    }

def filter_transaction_data(
    state: State,
    original_path: str,
    amount_min: float = None,
    amount_max: float = None,
    balance_min: float = None,
    balance_max: float = None,
    start_time: str = None,
    end_time: str = None
):
    """
    交易条件筛选工具。
    此工具用于根据用户指定的交易金额、余额、时间范围进行数据过滤和可视化展示。

    Param:
        - state: 当前状态
        - original_path: 原始数据路径
        - amount_min: 交易金额下限（可选）
        - amount_max: 交易金额上限（可选）
        - balance_min: 交易余额下限（可选）
        - balance_max: 交易余额上限（可选）
        - start_time: 交易时间起始（字符串，格式: 'yyyy-MM-dd HH:mm:ss'）
        - end_time: 交易时间结束（字符串，格式: 'yyyy-MM-dd HH:mm:ss'）

    return: 返回值是一个json结构
        data_path: str: 筛选后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv
        table_type: str: 表格类型，目前支持dataframe
        chart_type: str: 图表类型，目前支持image
        chart_path: str: 图表文件路径
        tips: str: 分析提示
        requirement: str: 可展示的格式化结果
    """
    print("start tool: filter_transaction_data")

    # 加载原始数据
    df = pd.read_csv(original_path)

    # 字段统一格式
    df['交易时间'] = pd.to_datetime(df['交易时间'], format='%Y-%m-%d %H:%M:%S')

    # 条件过滤
    if amount_min is not None:
        df = df[df['交易金额'] >= amount_min]
    if amount_max is not None:
        df = df[df['交易金额'] <= amount_max]
    if balance_min is not None:
        df = df[df['交易余额'] >= balance_min]
    if balance_max is not None:
        df = df[df['交易余额'] <= balance_max]
    if start_time is not None:
        df = df[df['交易时间'] >= pd.to_datetime(start_time)]
    if end_time is not None:
        df = df[df['交易时间'] <= pd.to_datetime(end_time)]

    # 保存结果
#    filtered_path = f"temp_data/filtered_data_{uuid.uuid4()}.csv"
    filtered_path = os.path.join(graph_data_dir, f"filtered_data_{uuid.uuid4()}.csv")
    df.to_csv(filtered_path, index=False)

    # 简单可视化（交易金额随时间分布图）
    import matplotlib.pyplot as plt
    # 加载中文字体
    font_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'STKAITI.TTF')
    zh_font = FontProperties(fname=font_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['交易时间'], df['交易金额'], color='blue', marker='o', markersize=5)
    plt.title("交易金额随时间分布", fontproperties=zh_font)
    plt.xlabel("交易时间", fontproperties=zh_font)
    plt.ylabel("交易金额", fontproperties=zh_font)
    plt.xticks(fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font)
    plt.grid(True)
#    chart_path = f"temp_data/filtered_chart_{uuid.uuid4()}.png"
    chart_path = os.path.join(graph_data_dir, f"filtered_chart_{uuid.uuid4()}.png")
    plt.savefig(chart_path)
    plt.close()

    # 格式化描述
    tip_lines = []
    if amount_min is not None or amount_max is not None:
        tip_lines.append(f"筛选条件：交易金额在 {amount_min or '-∞'} 到 {amount_max or '∞'} 之间")
    if balance_min is not None or balance_max is not None:
        tip_lines.append(f"筛选条件：交易余额在 {balance_min or '-∞'} 到 {balance_max or '∞'} 之间")
    if start_time is not None or end_time is not None:
        tip_lines.append(f"筛选条件：交易时间在 {start_time or '-∞'} 到 {end_time or '∞'} 之间")

    tips = "；".join(tip_lines) if tip_lines else "未提供筛选条件"

    return {
        "data_path": filtered_path,
        "data_type": "csv",
        "table_type": "dataframe",
        "chart_type": "image",
        "chart_path": chart_path,
        "tips": tips,
        "requirement": f"筛选后的交易记录共计 {len(df)} 条，结果已生成图表及表格。"
    }

def serialize_digraph_to_json(G):
    """
    将 NetworkX 有向图 (DiGraph) 转换为可 JSON 序列化的字典结构。

    返回格式:
    {
        "directed": True,
        "nodes": [{"id": "node1"}, {"id": "node2"}, ...],
        "edges": [{"source": "node1", "target": "node2", "amount": 100, "count": 1}, ...]
    }
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("输入必须是 nx.DiGraph 类型")

    # 获取所有边并按amount降序排序
    all_edges = sorted(G.edges(data=True), 
                    key=lambda x: x[2].get('amount', 0), 
                    reverse=True)

    # 取amount最大的前20条边
    top_edges = all_edges[:20]

    # 从这些边中提取涉及的节点（去重）
    top_nodes = set()
    for u, v, _ in top_edges:
        top_nodes.add(u)
        top_nodes.add(v)

    # 序列化节点（只包含top_nodes中的节点）
    nodes = [{"id": str(node), **G.nodes[node]} for node in top_nodes]

    # 序列化边（只包含top_edges中的边）
    edges = []
    for u, v, data in top_edges:
        edge_data = {"source": str(u), "target": str(v), **data}
        edges.append(edge_data)

    return {
        "directed": True,
        "nodes": nodes,
        "edges": edges
    }

def analyze_amount_networks(state: State):
    """
    金额网络分析工具
    此工具是团伙获利综合分析的一部分，也可以单独使用。
    Param:
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）和 json
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）和 json
        chart_type: str: 目前支持image（图片）
        chart_path: str: 如果chart_type为image，则chart_path为生成的图片路径
        html_path: str: 如果chart_type为network_graph，则html_path为生成的html路径
        tips: str: 分析提示
        requirement: str: 格式化后的分析结果，可直接用于展示，或供大模型做进一步分析
    """
    print("start tool: analyze_amount_networks")
    df = get_current_data(state)
    results = build_amount_networks(df)

    results_serialized = deepcopy(results)

    results_serialized["graph"] = serialize_digraph_to_json(results["graph"])

    results_serialized_temp = deepcopy(results_serialized)
    del results_serialized_temp["graph"]
#    output_path = f"temp_data/network_{uuid.uuid4()}.json"
    output_path = os.path.join(graph_data_dir, f"network_{uuid.uuid4()}.json")
    with open(output_path, 'w') as f:
        json.dump(results_serialized_temp, f)
    del results_serialized_temp

    # 生成网络图
    print("44444444444")
    html_path, _, png_path, _ = create_network_graph(results)
    print("555555555555555")

    return {
        "data_path": output_path,
        "data_type": "json",
        "table_type": "json",
        "chart_type": "image",
        "chart_path": png_path,
        "html_path": html_path,
        "tips": "交易网络分析已完成",
        "requirement": format_network_results(results)
    }

def analyze_balance_anomalies(state: State):
    """
    余额异常分析工具
    此工具是团伙获利综合分析的一部分，也可以单独使用。
    Param:
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）和 json
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）和 json
        chart_type: str: 目前支持image（图片）
        chart_path: str: 如果chart_type为image，则chart_path为生成的图片路径
        tips: str: 分析提示
        requirement: str: 格式化后的分析结果，可直接用于展示，或供大模型做进一步分析
    """
    print("start tool: analyze_balance_anomalies")
    df = get_current_data(state)
    results = detect_balance_anomalies(df)

#    output_path = f"temp_data/balance_{uuid.uuid4()}.json"
    output_path = os.path.join(graph_data_dir, f"balance_{uuid.uuid4()}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"in analyze_balance_anomalies, results: {results}")
    fig = generate_balance_visualization(results)
#    viz_path = f"temp_data/balance_viz_{uuid.uuid4()}.png"
    viz_path = os.path.join(graph_data_dir, f"balance_viz_{uuid.uuid4()}.png")
    fig.savefig(viz_path)

    return {
        "data_path": output_path,
        "data_type": "json",
        "table_type": "json",
        "chart_type": "image",
        "chart_path": viz_path,
        "tips": "余额异常分析已完成",
        "requirement": format_balance_results(results)
    }

def analyze_transaction_cycles(state: State):
    """
    交易循环分析工具
    此工具是团伙获利综合分析的一部分，也可以单独使用。
    Param:
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）和 json
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）和 json
        chart_type: str: 目前支持image（图片）
        chart_path: str: 如果chart_type为image，则chart_path为生成的图片路径
        tips: str: 分析提示
        requirement: str: 格式化后的分析结果，可直接用于展示，或供大模型做进一步分析
    """
    print("start tool: analyze_transaction_cycles")
    df = get_current_data(state)
    results = detect_transaction_cycles(df)

#    output_path = f"temp_data/cycles_{uuid.uuid4()}.json"
    output_path = os.path.join(graph_data_dir, f"cycles_{uuid.uuid4()}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f)

    fig = generate_cycle_visualization(results)
#    viz_path = f"temp_data/cycles_viz_{uuid.uuid4()}.png"
    viz_path = os.path.join(graph_data_dir, f"cycles_viz_{uuid.uuid4()}.png")
    fig.savefig(viz_path)

    return {
        "data_path": output_path,
        "data_type": "json",
        "table_type": "json",
        "chart_type": "image",
        "chart_path": viz_path,
        "tips": "交易循环分析已完成",
        "requirement": format_cycle_results(results)
    }

# ================ 辅助函数 ================

def get_current_data(state: State) -> pd.DataFrame:
    """获取当前数据集"""
    if state.get("current_df") is not None:
        return state["current_df"]
    elif state.get("origin_datapath"):
        state["current_df"] = pd.read_csv(state["origin_datapath"])
        return state["current_df"]
    else:
        raise ValueError("没有可用的数据")

# ================ 综合分析方法 ================
def comprehensive_analysis(state: State):
    """
    综合团伙获利分析。
    结合时间模式分析、交易网络分析、余额异常分析、交易循环分析，综合分析团伙获利风险。
    Param:
        - state: 当前状态
    return: 返回值是一个json结构
        data_path: str: 提取后的交易数据存储路径，以csv格式存储
        data_type: str: 数据类型，目前支持csv（表格数据）和 json
        table_type: str: 表格类型，目前支持dataframe（pandas dataframe）和 json
        chart_type: str: 目前支持image（图片）
        tips: str: 分析提示
        requirement: str: 格式化后的分析结果，可直接用于展示，或供大模型做进一步分析
    """
    print("start tool: comprehensive_analysis")
    df = get_current_data(state)

    # 执行所有分析
    time_results = detect_time_anomalies(df)
    network_results = build_amount_networks(df)
    network_results['graph'] = None
    balance_results = detect_balance_anomalies(df)
    cycle_results = detect_transaction_cycles(df)

    # 整合结果
    combined = {
        "time_analysis": time_results,
        "network_analysis": network_results,
        "balance_analysis": balance_results,
        "cycle_analysis": cycle_results,
        "risk_score": calculate_risk_score(time_results, network_results, balance_results, cycle_results)
    }

    print(f"combined in comprehensive_analysis: {combined}")

    # 保存结果
#    output_path = f"temp_data/comprehensive_{uuid.uuid4()}.json"
    output_path = os.path.join(graph_data_dir, f"comprehensive_{uuid.uuid4()}.json")
    with open(output_path, 'w') as f:
        try:
            json.dump(safe_json_dump(combined), f)
        except Exception as e:
            print(f"json.dump error in comprehensive_analysis: {e}")

    print(f"before generate_comprehensive_report: {combined}")
    # 生成综合报告
    report = generate_comprehensive_report(combined)

    return {
        "data_path": output_path,
        "data_type": "json",
        "table_type": "json",
        "chart_type": "composite",
        "tips": "综合团伙分析已完成",
        "requirement": report
    }



tools = [get_transaction_by_name, get_customer_list, get_knowledgebase_data, 
         analysis_gang_profits_with_data_fragment, analysis_gang_profits_with_key_info,
         comprehensive_analysis,  # 综合分析
         analyze_time_patterns,   # 增强分析方法
         analyze_amount_networks,
         analyze_balance_anomalies,
         analyze_transaction_cycles,
         filter_transaction_data]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
#    prompt = f"""
#    你是一个agent系统，通过绑定的工具集完成任务。
#    你只能做工具集中提供的的功能，其他功能都不能做。
#    不要在回应中提及你的名字，否则会被认为是广告。
#    不要在回应中提及本说明中提到的额外信息，例如：您已经提供了一个数据路径，否则会造成信息泄露。
#    不要在回应中给用户任何建议，否则会被认为是广告。
#    这里提供了一些额外的信息：{str(state["messages"][-1].additional_kwargs)}。
#    """
    prompt = f"""
    你是一个agent系统，通过绑定的工具集完成任务。
    不要在回应中给用户任何建议，否则会被认为是广告。
    这里提供了一些额外的信息：{str(state["messages"][-1].additional_kwargs)}。
    """
    system_message = SystemMessage(
        content=prompt,
    )
    human_message = HumanMessage(
        content=state["messages"][-1].content
    )
    return {"messages": [llm_with_tools.invoke([system_message, human_message])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

import gradio as gr
import pandas as pd
import zipfile
import os
import json
import time
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
import uuid
from data_analysis_with_agent.backend.app.knowledgebase import create_knowledge_graph


# 设置浏览器为 Windows 的 start 命令
os.environ["BROWSER"] = "cmd.exe /c start"

def add_system_message(message, progress=gr.Progress()):
    progress(0, desc=message)  # 显示进度
    time.sleep(0.5)  # 为了让用户能看到处理过程

def process_uploaded_file(zip_file):
    chat_history = []

#    temp_path = f"temp_data/{str(uuid.uuid4())}"
    temp_path = os.path.join(graph_data_dir, f"{str(uuid.uuid4())}")
    os.makedirs(temp_path, exist_ok=True)

    # 1. 解压文件
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 开始解压文件...")
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        chat_history = add_system_message(f"找到 {len(file_list)} 个文件")
        # 逐个文件解压并更新状态
        for i, file in enumerate(file_list):
            zip_ref.extract(file, temp_path)
            if i % 5 == 0 or i == len(file_list)-1:  # 每5个文件或最后一个文件更新一次
                add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 已解压 {i+1}/{len(file_list)} 个文件")
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 解压完成")

    # 2. 加载CSV文件
    dfs = []
    csv_files = [f for f in os.listdir(temp_path) if f.lower().endswith('.csv')]
    add_system_message(f"开始加载 {len(csv_files)} 个CSV文件...")

    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(os.path.join(temp_path, file))
            dfs.append(df)
            add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 已加载: {file} ({len(df)}行)")
        except Exception as e:
            add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 加载失败: {file} - {str(e)}")

    if not dfs:
        add_system_message("错误: 没有找到可用的CSV数据")
        return chat_history, pd.DataFrame(), None

    chat_history = []
    # 3. 合并数据
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 开始合并数据...")
    combined_df = pd.concat(dfs)
    add_system_message(f"合并后总数据量: {len(combined_df)} 行")

    # 4. 数据清洗
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 开始数据清洗...")
    cleaned_df = combined_df.dropna(how="all").drop_duplicates()
    removed_rows = len(combined_df) - len(cleaned_df)
    add_system_message(f"清洗完成，移除了 {removed_rows} 行，剩余 {len(cleaned_df)} 行有效数据")

    # 5. 生成统计信息
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 正在生成统计信息...")
    stats = {
        "total_records": len(cleaned_df),
        "unique_counts": {col: cleaned_df[col].nunique() for col in cleaned_df.columns}
    }

    # 6. 创建图表
    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 正在生成可视化图表...")
    # 组织数据
    rows = [
        ["全部", stats["total_records"]]
    ] + [[col, count] for col, count in stats["unique_counts"].items()]

    # 创建 DataFrame
    data = pd.DataFrame(rows, columns=["字段", "行数"])

    # 创建 Plotly 柱状图
    fig = px.bar(data, x="字段", y="行数", title="数据统计",
                 labels={"字段": "字段", "行数": "行数"}, text="行数")

    fig.update_traces(marker_color="blue", textposition="outside")
    fig.update_layout(hovermode="x unified")

    add_system_message(f"{datetime.now().strftime('%H:%M:%S')} 数据处理完成!")

    file_list = [temp_path + "/" + f for f in file_list]

    control_table = gr.update(visible=True, value=cleaned_df.head(10))
    control_chart_plotly = gr.update(visible=True, value=fig)
    return chat_history, control_table, control_chart_plotly, file_list

def empty_plot():
    fig, ax = plt.subplots()
    return fig

# 模拟大模型分析
def analyze_request(query, chat_history, df, file_list):
    empty_df = pd.DataFrame()
    empty_chart = empty_plot()
    empty_json = {}
    empty_image = None
    control_empty_table_csv = gr.update(visible=True, value=empty_df)
    control_empty_chart_plotly = gr.update(visible=False, value=empty_chart)
    control_empty_table_json = gr.update(visible=False, value=empty_json)
    control_empty_image = gr.update(visible=False, value=empty_image)

    file_path = ""
    if len(file_list) > 0:
        file_path = file_list[-1]

    chat_history.append({"role": "user", "content": query})
    yield chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image

    bot_message = "大模型在思考中......"
    chat_history.append({"role": "assistant", "content": bot_message})
    yield chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image

    message = HumanMessage(
        content=query,
        additional_kwargs={
            "origin_datapath": file_path
        }
    )
    events = graph.stream(
        {"messages": [message]}, stream_mode="messages"
    )

    tool_message = ""
    agent_state = "preparing"
    for event in events:
        if agent_state == "processing_ready":
            agent_state = "processing"

        if isinstance(event[0], ToolMessage):
            tool_message = event[0].content
        elif isinstance(event[0], AIMessageChunk):
            tool_calls = event[0].tool_calls
            response_metadata = event[0].response_metadata
            if agent_state == "preparing" and tool_calls and len(tool_calls) > 0:
                agent_state = "ready"
                bot_message += "\n智能体函数准备中..."
                chat_history[-1] = {"role": "assistant", "content": bot_message}
                yield chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image
                continue

            if agent_state == "ready" and response_metadata.get("finish_reason", "") == "tool_calls":
                agent_state = "processing_ready"
                bot_message += "\n智能体函数处理中..."
                chat_history[-1] = {"role": "assistant", "content": bot_message}
                yield chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image

            if event[0].content:
                bot_message += event[0].content
                chat_history[-1] = {"role": "assistant", "content": bot_message}
                yield chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image

    print(f"tool_message: {tool_message}")

    tool_retval = {}
    if tool_message:
        try:
            tool_retval = json.loads(tool_message)
        except:
            bot_message += tool_message
            chat_history[-1] = {"role": "assistant", "content": bot_message}
            return chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image

    if not tool_retval:
        return chat_history, control_empty_table_csv, control_empty_chart_plotly, control_empty_table_json, control_empty_image
    else:
        data_path = tool_retval["data_path"]
        data_type = tool_retval["data_type"]    # csv, json
        table_type = tool_retval["table_type"]  # dataframe, json
        chart_type = tool_retval["chart_type"]  # bar, network_graph, image
        ret_str = tool_retval.get("requirement", "")
        chart_path = tool_retval.get("chart_path", "")

        table_csv_value = empty_df
        table_json_value = {}

        control_table_dataframe = gr.update(visible=False, value=empty_df)
        control_table_json = gr.update(visible=False, value={})
        control_chart_plotly = gr.update(visible=False, value=empty_chart)
        control_image = gr.update(visible=False, value=None)

        if data_type == "csv":
            table_csv_value = pd.read_csv(data_path)
            control_table_dataframe = gr.update(visible=True, value=table_csv_value)
        elif data_type == "json":
            with open(data_path, 'r') as file:
                table_json_value = json.load(file)
            control_table_json = gr.update(visible=True, value=table_json_value)
        else:
            control_table_dataframe = gr.update(visible=True, value=empty_df)


        if chart_type == "network_graph":
            # 生成唯一的输出文件名
            graph_id = str(uuid.uuid4())
            html_output = os.path.join(graph_data_dir, f"knowledge_graph_{graph_id}.html")
            html_output_path, html_gen_state, png_output_path, png_gen_state = create_knowledge_graph(data_path, html_output)
            print(f"html_output_path: {html_output_path}, html_gen_state: {html_gen_state}, png_output_path: {png_output_path}, png_gen_state: {png_gen_state}")
            if html_gen_state:
                bot_message += f"\n点击这里查看 <a href='{args.knowledge_graph_server_ip}{os.path.basename(html_output_path)}' target='_blank'><font color='blue'>资金交易知识图谱</font></a>"
                chat_history[-1] = {"role": "assistant", "content": bot_message}
            if png_gen_state:
                control_chart_plotly = gr.update(visible=True, value=png_output_path)
                try:
                    # 显示PNG图片版本的知识图谱
                    graph_img = plt.imread(png_output_path)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(graph_img)
                    ax.axis('off')
                    ax.set_title("交易数据知识图谱")
                    control_chart_plotly = gr.update(visible=True, value=fig)
                except Exception as e:
                    print(f"Error showing PNG graph: {str(e)}")
        elif chart_type == "image":
            # 显示柱状图
            control_image = gr.update(visible=True, value=chart_path)

        yield chat_history, control_table_dataframe, control_chart_plotly, control_table_json, control_image

with gr.Blocks(css=".gray-button { color: gray; }") as demo:
    state = gr.State(value=[])
    # 界面布局
    gr.Markdown("## 经侦案件数据分析系统")
   
    # 顶部上传区域
    with gr.Row(height="10%"):
        upload = gr.UploadButton("上传ZIP文件", file_types=[".zip"])

    # 中间区域
    with gr.Row(height="90%"):
        with gr.Column(scale=4):
            # Chatbot组件
            chatbot = gr.Chatbot(height=400, type="messages")
           
            # 优化输入框和按钮的布局
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="输入分析请求...",
                    show_label=False,
                    container=False,
                    scale=8
                )
                send_btn = gr.Button(
                    "发送",
                    scale=2,
                    min_width=80
                )
       
        with gr.Column(scale=6):
            data_table = gr.DataFrame(label="数据预览", interactive=True)
            json_display = gr.JSON(label="JSON预览", visible=False)
            chart = gr.Plot(label="数据分析图表")
            image = gr.Image(label="分析图表", visible=False)
    
    # ================ 更新Gradio界面 ================
    def update_ui_controls():
        """更新UI控件显示增强分析选项"""
        with gr.Accordion("查询统计", open=False):
            # 底部提示区
            with gr.Row(height="10%"):
                gr.Button("查看客户列表").click(
                    fn=lambda: "查看客户列表",
                    outputs=user_input
                )
                gr.Button("查看知识图谱").click(
                    fn=lambda: "查看知识图谱",
                    outputs=user_input
                )
                gr.Button("交易数据过滤").click(
                    fn=lambda: "根据交易时间、交易金额、交易余额进行数据过滤，请输入条件：",
                    outputs=user_input
                )
        with gr.Accordion("技术分析", open=False):
            with gr.Row():
                gr.Button("时间模式分析").click(
                    fn=lambda: "执行时间模式分析",
                    outputs=user_input
                )
                gr.Button("交易网络分析").click(
                    fn=lambda: "执行交易网络分析",
                    outputs=user_input
                )
                gr.Button("余额异常分析").click(
                    fn=lambda: "执行余额异常分析",
                    outputs=user_input
                )
                gr.Button("交易循环分析").click(
                    fn=lambda: "执行交易循环分析",
                    outputs=user_input
                )

        with gr.Accordion("业务分析", open=False):
            with gr.Row():
                gr.Button("团伙获利分析").click(
                    fn=lambda: "用{关键信息提取或数据分片}的方法做团伙获利分析。",
                    outputs=user_input
                )
                gr.Button("综合团伙分析").click(
                    fn=lambda: "执行综合团伙分析",
                    outputs=user_input
                )
    update_ui_controls()  # 添加高级分析控件

    # 事件处理
    upload.upload(
        process_uploaded_file,
        inputs=upload,
        outputs=[chatbot, data_table, chart, state]
    )

    send_btn.click(
        analyze_request,
        inputs=[user_input, chatbot, data_table, state],
        outputs=[chatbot, data_table, chart, json_display, image]
    )
    user_input.submit(
        analyze_request,
        inputs=[user_input, chatbot, data_table, state],
        outputs=[chatbot, data_table, chart, json_display, image]
    )

#if __name__ == "__main__":
allowed_paths = [graph_data_dir]
demo.launch(server_name=args.server_name, allowed_paths=allowed_paths)
