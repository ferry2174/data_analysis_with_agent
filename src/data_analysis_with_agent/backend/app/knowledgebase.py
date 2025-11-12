import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def compress_data(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding='utf-8')

    # 创建一个字典来存储压缩后的数据
    entities = {}

    # 创建全局关系网络
    global_network = {
        "客户关系网": defaultdict(set),  # 客户与交易对方的关系
        "IP关系网": defaultdict(set),    # IP地址与客户的关系
        "MAC关系网": defaultdict(set),   # MAC地址与客户的关系
        "账号关系网": defaultdict(set),  # 账号与客户的关系
        "卡号关系网": defaultdict(set),  # 卡号与客户的关系
        "交易金额分布": defaultdict(int), # 交易金额分布
    }

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 提取客户名称作为主键
        customer_name = row['客户名称'] if pd.notna(row['客户名称']) else ''

        # 跳过没有客户名称的数据
        if not customer_name:
            continue

        # 使用客户名称作为唯一标识符
        entity_id = customer_name

        # 提取实体信息
        query_card = row['查询卡号'] if pd.notna(row['查询卡号']) else ''
        query_account = row['查询账号'] if pd.notna(row['查询账号']) else ''
        counter_party = row['交易对方名称'] if pd.notna(row['交易对方名称']) else ''
        ip_address = row['本方IP地址'] if pd.notna(row['本方IP地址']) else ''
        mac_address = row['本方MAC地址'] if pd.notna(row['本方MAC地址']) else ''
        bank_name = row['银行'] if pd.notna(row['银行']) else ''
        transaction_type = row['交易类型'] if pd.notna(row['交易类型']) else ''

        # 获取借贷标志
        debit_credit_flag = row['借贷标志'] if pd.notna(row['借贷标志']) else ''

        # 提取关系信息
        transaction_amount = float(row['交易金额']) if pd.notna(row['交易金额']) else 0
        transaction_balance = float(row['交易余额']) if pd.notna(row['交易余额']) else 0
        transaction_time = row['交易时间'] if pd.notna(row['交易时间']) else ''

        # 如果实体不存在，则创建新实体
        if entity_id not in entities:
            entities[entity_id] = {
                "实体信息": {
                    "客户名称": customer_name,
                },
                "查询账号": set(),  # 使用集合存储所有不同的查询账号
                "查询卡号": set(),  # 使用集合存储所有不同的查询卡号
                "交易对方": set(),  # 使用集合存储所有不同的交易对方
                "IP地址": set(),    # 使用集合存储所有不同的IP地址
                "MAC地址": set(),   # 使用集合存储所有不同的MAC地址
                "银行": set(),      # 使用集合存储所有不同的银行
                "交易类型": set(),  # 使用集合存储所有不同的交易类型
                "交易记录": [],
                "月度交易": {},     # 用于按月统计交易次数
                "交易次数": 0,      # 统计交易次数
                "出账次数": 0,      # 统计出账次数
                "入账次数": 0,      # 统计入账次数
                "交易金额": {       # 按交易金额分级统计
                    "超过15000": 0,     # 超过15000
                    "9000-15000": 0,     # 9000-15000
                    "低于9000": 0      # 低于9000
                },
                "交易余额": {       # 按交易余额分级统计
                    "超过60000": 0,     # 超过60000
                    "20000-60000": 0,     # 20000-60000
                    "低于20000": 0      # 低于20000
                },
                "总交易金额": 0,    # 累计交易金额
                "借入总金额": 0,    # 累计借入金额
                "借出总金额": 0,    # 累计借出金额
                "交易对方关系": defaultdict(int),  # 与每个交易对方的交易次数
                "交易对方金额": defaultdict(float), # 与每个交易对方的交易金额
                "连续交易": [],     # 记录连续交易的时间间隔
                "最大单笔交易": 0,  # 最大单笔交易金额
                "最小单笔交易": float('inf'),  # 最小单笔交易金额
            }

        # 添加查询账号、查询卡号、交易对方、IP地址和MAC地址
        if query_account:
            entities[entity_id]["查询账号"].add(query_account)
            global_network["账号关系网"][query_account].add(customer_name)
        if query_card:
            entities[entity_id]["查询卡号"].add(query_card)
            global_network["卡号关系网"][query_card].add(customer_name)
        if counter_party:
            entities[entity_id]["交易对方"].add(counter_party)
            global_network["客户关系网"][customer_name].add(counter_party)
            entities[entity_id]["交易对方关系"][counter_party] += 1
            entities[entity_id]["交易对方金额"][counter_party] += transaction_amount
        if ip_address:
            entities[entity_id]["IP地址"].add(ip_address)
            global_network["IP关系网"][ip_address].add(customer_name)
        if mac_address:
            entities[entity_id]["MAC地址"].add(mac_address)
            global_network["MAC关系网"][mac_address].add(customer_name)
        if bank_name:
            entities[entity_id]["银行"].add(bank_name)
        if transaction_type:
            entities[entity_id]["交易类型"].add(transaction_type)

        # 添加交易信息
        transaction_info = {
            "金额": transaction_amount,
            "余额": transaction_balance,
            "时间": transaction_time,
            "对方": counter_party,
            "借贷标志": debit_credit_flag
        }
        entities[entity_id]["交易记录"].append(transaction_info)
        entities[entity_id]["交易次数"] += 1
        entities[entity_id]["总交易金额"] += transaction_amount

        # 更新最大和最小单笔交易
        entities[entity_id]["最大单笔交易"] = max(entities[entity_id]["最大单笔交易"], transaction_amount)
        if transaction_amount > 0:  # 只考虑正数金额
            entities[entity_id]["最小单笔交易"] = min(entities[entity_id]["最小单笔交易"], transaction_amount)

        # 根据借贷标志统计出入账次数和金额
        if debit_credit_flag == "出":
            entities[entity_id]["出账次数"] += 1
            entities[entity_id]["借出总金额"] += transaction_amount
        elif debit_credit_flag == "借" or debit_credit_flag == "进":
            entities[entity_id]["入账次数"] += 1
            entities[entity_id]["借入总金额"] += transaction_amount

        # 按交易金额分级统计
        if transaction_amount > 15000:
            entities[entity_id]["交易金额"]["超过15000"] += 1
            global_network["交易金额分布"]["超过15000"] += 1
        elif transaction_amount >= 9000:
            entities[entity_id]["交易金额"]["9000-15000"] += 1
            global_network["交易金额分布"]["9000-15000"] += 1
        else:
            entities[entity_id]["交易金额"]["低于9000"] += 1
            global_network["交易金额分布"]["低于9000"] += 1

        # 按交易余额分级统计
        if transaction_balance > 60000:
            entities[entity_id]["交易余额"]["超过60000"] += 1
        elif transaction_balance >= 20000:
            entities[entity_id]["交易余额"]["20000-60000"] += 1
        else:
            entities[entity_id]["交易余额"]["低于20000"] += 1

        # 按月统计交易
        if transaction_time:
            try:
                # 解析交易时间
                transaction_date = datetime.strptime(transaction_time, "%Y-%m-%d %H:%M:%S")

                # 按月统计
                month_key = f"{transaction_date.year}-{transaction_date.month:02d}"
                if month_key not in entities[entity_id]["月度交易"]:
                    entities[entity_id]["月度交易"][month_key] = 0
                entities[entity_id]["月度交易"][month_key] += 1

                # 计算连续交易的时间间隔
                if len(entities[entity_id]["交易记录"]) > 1:
                    prev_transaction = entities[entity_id]["交易记录"][-2]
                    if "时间" in prev_transaction and prev_transaction["时间"]:
                        try:
                            prev_date = datetime.strptime(prev_transaction["时间"], "%Y-%m-%d %H:%M:%S")
                            time_diff = (transaction_date - prev_date).total_seconds() / 60  # 时间差（分钟）
                            entities[entity_id]["连续交易"].append(time_diff)
                        except ValueError:
                            pass
            except ValueError:
                # 如果日期格式不正确，跳过
                pass

    # 处理连续交易数据，计算平均间隔和异常快速交易
    for entity_id, data in entities.items():
        if data["连续交易"]:
            data["平均交易间隔"] = sum(data["连续交易"]) / len(data["连续交易"])
            data["快速连续交易"] = sum(1 for interval in data["连续交易"] if interval < 5)  # 5分钟内的连续交易
        else:
            data["平均交易间隔"] = 0
            data["快速连续交易"] = 0

        # 如果最小单笔交易仍为初始值，设为0
        if data["最小单笔交易"] == float('inf'):
            data["最小单笔交易"] = 0

    # 创建最终的JSON结构
    result = {
        "实体列表": [],
        "关系网络": {
            "共用IP地址的客户": [],
            "共用MAC地址的客户": [],
            "共用账号的客户": [],
            "共用卡号的客户": [],
            "频繁交易对方": [],
            "交易金额分布": dict(global_network["交易金额分布"]),
        }
    }

    # 处理共用关系
    for ip, customers in global_network["IP关系网"].items():
        if len(customers) > 1:  # 如果有多个客户共用同一IP
            result["关系网络"]["共用IP地址的客户"].append({
                "IP地址": ip,
                "客户列表": list(customers)
            })

    for mac, customers in global_network["MAC关系网"].items():
        if len(customers) > 1:  # 如果有多个客户共用同一MAC
            result["关系网络"]["共用MAC地址的客户"].append({
                "MAC地址": mac,
                "客户列表": list(customers)
            })

    for account, customers in global_network["账号关系网"].items():
        if len(customers) > 1:  # 如果有多个客户共用同一账号
            result["关系网络"]["共用账号的客户"].append({
                "账号": account,
                "客户列表": list(customers)
            })

    for card, customers in global_network["卡号关系网"].items():
        if len(customers) > 1:  # 如果有多个客户共用同一卡号
            result["关系网络"]["共用卡号的客户"].append({
                "卡号": card,
                "客户列表": list(customers)
            })

    # 找出频繁交易的对方
    counter_party_freq = Counter()
    for entity_id, data in entities.items():
        for counter_party, count in data["交易对方关系"].items():
            counter_party_freq[counter_party] += count

    # 添加前10个最频繁交易对方
    for counter_party, count in counter_party_freq.most_common(10):
        result["关系网络"]["频繁交易对方"].append({
            "交易对方": counter_party,
            "交易次数": count
        })

    # 将实体数据添加到结果中
    for entity_id, data in entities.items():
        # 找出最频繁交易的对方
        top_counter_parties = sorted(data["交易对方关系"].items(), key=lambda x: x[1], reverse=True)[:5]

        entity_data = {
            "实体信息": data["实体信息"],
            "账号信息": {
                "查询账号列表": list(data["查询账号"]),
                "查询账号数量": len(data["查询账号"]),
                "查询卡号列表": list(data["查询卡号"]),
                "查询卡号数量": len(data["查询卡号"]),
                "银行列表": list(data["银行"]),
                "银行数量": len(data["银行"]),
            },
            "网络信息": {
                "IP地址列表": list(data["IP地址"]),
                "IP地址数量": len(data["IP地址"]),
                "MAC地址列表": list(data["MAC地址"]),
                "MAC地址数量": len(data["MAC地址"]),
            },
            "交易统计": {
                "交易次数": data["交易次数"],
                "出账次数": data["出账次数"],
                "入账次数": data["入账次数"],
                "总交易金额": data["总交易金额"],
                "借入总金额": data["借入总金额"],
                "借出总金额": data["借出总金额"],
                "最大单笔交易": data["最大单笔交易"],
                "最小单笔交易": data["最小单笔交易"],
                "平均交易间隔(分钟)": data["平均交易间隔"],
                "快速连续交易次数": data["快速连续交易"],
                "交易类型": list(data["交易类型"]),
                "交易对方列表": list(data["交易对方"]),
                "交易对方数量": len(data["交易对方"]),
                "最频繁交易对方": [{"对方": cp, "次数": count} for cp, count in top_counter_parties],
                "月度交易次数": data["月度交易"],
                "交易金额分级统计": data["交易金额"],
                "交易余额分级统计": data["交易余额"]
            }
        }

        result["实体列表"].append(entity_data)

    # 将压缩后的数据写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def create_knowledge_graph(json_file, output_html="transaction_knowledge_graph.html"):
    """
    Create a knowledge graph from the compressed transaction data JSON file.

    Args:
        json_file (str): Path to the JSON file containing compressed transaction data
        output_html (str): Path to save the interactive HTML visualization

    Returns:
        G (networkx.Graph): The generated knowledge graph
    """
    # Load the JSON data
    if isinstance(json_file, str):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json_file

    # Create a graph
    G = nx.Graph()

    # Add entity nodes (customers)
    for entity in data["实体列表"]:
        customer_name = entity["实体信息"]["客户名称"]

        # Calculate node size based on transaction count (scaled for visibility)
        node_size = 15 + entity["交易统计"]["交易次数"] // 100

        # Create detailed tooltip with customer information
        tooltip = f"""
        <b>客户: {customer_name}</b><br>
        <b>交易统计:</b><br>
        - 交易次数: {entity['交易统计']['交易次数']}<br>
        - 总交易金额: {entity['交易统计']['总交易金额']:.2f}<br>
        - 出账次数: {entity['交易统计']['出账次数']}<br>
        - 入账次数: {entity['交易统计']['入账次数']}<br>
        - 最大单笔交易: {entity['交易统计']['最大单笔交易']:.2f}<br>
        - 快速连续交易: {entity['交易统计']['快速连续交易次数']}<br>
        <b>账号信息:</b><br>
        - 查询账号: {', '.join(str(acc) for acc in entity['账号信息']['查询账号列表'])}<br>
        - 查询卡号: {', '.join(str(card) for card in entity['账号信息']['查询卡号列表'])}<br>
        <b>网络信息:</b><br>
        - IP地址数量: {entity['网络信息']['IP地址数量']}<br>
        - MAC地址数量: {entity['网络信息']['MAC地址数量']}<br>
        """

        G.add_node(customer_name, 
                   type="customer", 
                   size=node_size,
                   title=tooltip)

        # Add IP address nodes and connections
        for ip in entity["网络信息"]["IP地址列表"]:
            ip_node_id = f"IP:{ip}"
            if not G.has_node(ip_node_id):
                G.add_node(ip_node_id, 
                           type="ip", 
                           size=8,
                           title=f"IP地址: {ip}")

            G.add_edge(customer_name, 
                       ip_node_id, 
                       relationship="使用IP",
                       title="使用IP地址")

        # Add MAC address nodes and connections
        for mac in entity["网络信息"]["MAC地址列表"]:
            mac_node_id = f"MAC:{mac}"
            if not G.has_node(mac_node_id):
                G.add_node(mac_node_id, 
                           type="mac", 
                           size=8,
                           title=f"MAC地址: {mac}")

            G.add_edge(customer_name, 
                       mac_node_id, 
                       relationship="使用MAC",
                       title="使用MAC地址")

        # Add account nodes and connections
        for account in entity["账号信息"]["查询账号列表"]:
            account_node_id = f"账号:{account}"
            if not G.has_node(account_node_id):
                G.add_node(account_node_id, 
                           type="account", 
                           size=8,
                           title=f"账号: {account}")

            G.add_edge(customer_name, 
                       account_node_id, 
                       relationship="拥有账号",
                       title="拥有账号")

        # Add card nodes and connections
        for card in entity["账号信息"]["查询卡号列表"]:
            card_node_id = f"卡号:{card}"
            if not G.has_node(card_node_id):
                G.add_node(card_node_id, 
                           type="card", 
                           size=8,
                           title=f"卡号: {card}")

            G.add_edge(customer_name, 
                       card_node_id, 
                       relationship="拥有卡",
                       title="拥有卡")

        # Add transaction counterparties and connections
        for counterparty in entity["交易统计"]["最频繁交易对方"]:
            counterparty_name = counterparty["对方"]
            counterparty_node_id = f"对方:{counterparty_name}"

            if not G.has_node(counterparty_node_id):
                G.add_node(counterparty_node_id, 
                           type="counterparty", 
                           size=7,
                           title=f"交易对方: {counterparty_name}")

            # Add edge with transaction count as weight
            G.add_edge(customer_name, 
                       counterparty_node_id, 
                       weight=counterparty["次数"],
                       relationship="交易",
                       title=f"交易次数: {counterparty['次数']}")

    # Add shared resource relationships from the network data
    for ip_data in data["关系网络"]["共用IP地址的客户"]:
        ip = ip_data["IP地址"]
        customers = ip_data["客户列表"]

        # Add IP node if it doesn't exist
        ip_node_id = f"IP:{ip}"
        if not G.has_node(ip_node_id):
            G.add_node(ip_node_id, 
                       type="ip", 
                       size=10,
                       title=f"IP地址: {ip}<br>共用客户: {', '.join(customers)}")

        # Add edges between all customers and this IP
        for customer in customers:
            if G.has_node(customer) and not G.has_edge(customer, ip_node_id):
                G.add_edge(customer, 
                           ip_node_id, 
                           relationship="共用IP",
                           title="共用IP地址")

        # Add edges between all pairs of customers sharing this IP
        for i in range(len(customers)):
            for j in range(i+1, len(customers)):
                customer1 = customers[i]
                customer2 = customers[j]

                # Skip if either node doesn't exist
                if not (G.has_node(customer1) and G.has_node(customer2)):
                    continue

                # Add or update edge between customers
                if G.has_edge(customer1, customer2):
                    # If edge exists, update relationship info
                    current_title = G[customer1][customer2].get('title', '')
                    if "共用IP" not in current_title:
                        G[customer1][customer2]['title'] = f"{current_title}, 共用IP"
                    G[customer1][customer2]['weight'] = G[customer1][customer2].get('weight', 1) + 1
                else:
                    # Create new edge
                    G.add_edge(customer1, 
                               customer2, 
                               weight=2, 
                               title="共用IP",
                               relationship="共用IP")

    # Add MAC address relationships
    for mac_data in data["关系网络"]["共用MAC地址的客户"]:
        mac = mac_data["MAC地址"]
        customers = mac_data["客户列表"]

        # Add MAC node if it doesn't exist
        mac_node_id = f"MAC:{mac}"
        if not G.has_node(mac_node_id):
            G.add_node(mac_node_id, 
                       type="mac", 
                       size=10,
                       title=f"MAC地址: {mac}<br>共用客户: {', '.join(customers)}")

        # Add edges between all customers and this MAC
        for customer in customers:
            if G.has_node(customer) and not G.has_edge(customer, mac_node_id):
                G.add_edge(customer, 
                           mac_node_id, 
                           relationship="共用MAC",
                           title="共用MAC地址")

        # Add edges between all pairs of customers sharing this MAC
        for i in range(len(customers)):
            for j in range(i+1, len(customers)):
                customer1 = customers[i]
                customer2 = customers[j]

                # Skip if either node doesn't exist
                if not (G.has_node(customer1) and G.has_node(customer2)):
                    continue

                # Add or update edge between customers
                if G.has_edge(customer1, customer2):
                    # If edge exists, update relationship info
                    current_title = G[customer1][customer2].get('title', '')
                    if "共用MAC" not in current_title:
                        G[customer1][customer2]['title'] = f"{current_title}, 共用MAC"
                    G[customer1][customer2]['weight'] = G[customer1][customer2].get('weight', 1) + 1
                else:
                    # Create new edge
                    G.add_edge(customer1, 
                               customer2, 
                               weight=2, 
                               title="共用MAC",
                               relationship="共用MAC")

    # Add frequent transaction counterparties
    for counterparty_data in data["关系网络"]["频繁交易对方"]:
        counterparty_name = counterparty_data["交易对方"]
        transaction_count = counterparty_data["交易次数"]

        # Add counterparty node if it doesn't exist
        counterparty_node_id = f"对方:{counterparty_name}"
        if not G.has_node(counterparty_node_id):
            # Size based on transaction count
            size = 5 + min(transaction_count // 100, 15)
            G.add_node(counterparty_node_id, 
                       type="counterparty", 
                       size=size,
                       title=f"交易对方: {counterparty_name}<br>总交易次数: {transaction_count}")

    # Create interactive visualization
    return create_interactive_graph(G, output_html)

def create_interactive_graph(G, output_file):
    """
    Create an interactive HTML visualization of the knowledge graph

    Args:
        G (networkx.Graph): The graph to visualize
        output_file (str): Path to save the HTML file
    """
    html_generate_state = False
    png_generate_state = False
    try:
    # Create a custom HTML template with additional features
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>金融交易知识图谱</title>
            <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vis-network@9.1.2/dist/vis-network.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/vis-network@9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
            <style type="text/css">
                #mynetwork {
                    width: 100%;
                    height: 90vh;
                    background-color: #222222;
                    border: 1px solid lightgray;
                    position: relative;
                }
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #333333;
                    color: white;
                }
                .container {
                    width: 100%;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                .header {
                    padding: 10px;
                    background-color: #444444;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .controls {
                    display: flex;
                    gap: 10px;
                }
                button {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 14px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                .legend {
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background-color: rgba(0, 0, 0, 0.7);
                    padding: 10px;
                    border-radius: 5px;
                    z-index: 1000;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin-bottom: 5px;
                }
                .legend-color {
                    width: 15px;
                    height: 15px;
                    margin-right: 5px;
                    border-radius: 50%;
                }
                .search-container {
                    position: relative;
                    margin-right: 10px;
                }
                #search-input {
                    padding: 8px;
                    border-radius: 4px;
                    border: none;
                    width: 200px;
                }
                #search-results {
                    position: absolute;
                    top: 100%;
                    left: 0;
                    width: 100%;
                    max-height: 200px;
                    overflow-y: auto;
                    background-color: #444444;
                    border-radius: 4px;
                    display: none;
                    z-index: 1000;
                }
                .search-result {
                    padding: 8px;
                    cursor: pointer;
                }
                .search-result:hover {
                    background-color: #555555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>金融交易知识图谱</h2>
                    <div class="controls">
                        <div class="search-container">
                            <input type="text" id="search-input" placeholder="搜索节点...">
                            <div id="search-results"></div>
                        </div>
                        <button onclick="resetView()">重置视图</button>
                        <button onclick="togglePhysics()">切换物理引擎</button>
                    </div>
                </div>
                <div id="mynetwork"></div>
                <div class="legend">
                    <h3>图例</h3>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #4CAF50;"></div>
                        <span>客户</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #2196F3;"></div>
                        <span>交易对方</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF5722;"></div>
                        <span>IP地址</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9C27B0;"></div>
                        <span>MAC地址</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FFEB3B;"></div>
                        <span>账号</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #00BCD4;"></div>
                        <span>卡号</span>
                    </div>
                </div>
            </div>

            <script type="text/javascript">
                // 创建节点和边的数据集
                var nodes = new vis.DataSet(NODES_DATA);
                var edges = new vis.DataSet(EDGES_DATA);

                // 创建网络图容器
                var container = document.getElementById('mynetwork');

                // 配置网络图选项
                var options = {
                    nodes: {
                        shape: 'dot',
                        scaling: {
                            min: 10,
                            max: 30,
                            label: {
                                enabled: true,
                                min: 14,
                                max: 30,
                                maxVisible: 30,
                                drawThreshold: 5
                            },
                            customScalingFunction: function (min, max, total, value) {
                                if (max === min) {
                                    return 0.5;
                                } else {
                                    let scale = 1 / (max - min);
                                    return Math.max(0, (value - min) * scale);
                                }
                            }
                        },
                        font: {
                            size: 14,
                            face: 'Arial',
                            color: 'white'
                        }
                    },
                    edges: {
                        width: 2,
                        smooth: {
                            type: 'continuous'
                        },
                        arrows: {
                            to: {
                                enabled: false
                            }
                        },
                        color: {
                            inherit: false
                        },
                        font: {
                            size: 12,
                            face: 'Arial',
                            color: 'white'
                        }
                    },
                    physics: {
                        stabilization: false,
                        barnesHut: {
                            gravitationalConstant: -80000,
                            centralGravity: 0.3,
                            springLength: 95,
                            springConstant: 0.04,
                            damping: 0.09,
                            avoidOverlap: 0.1
                        },
                        maxVelocity: 50,
                        minVelocity: 0.1,
                        solver: 'barnesHut',
                        timestep: 0.5,
                        adaptiveTimestep: true
                    },
                    interaction: {
                        tooltipDelay: 200,
                        hideEdgesOnDrag: true,
                        hover: true,
                        multiselect: true,
                        navigationButtons: true,
                        keyboard: true
                    },
                    groups: {
                        customer: {
                            color: {
                                background: '#4CAF50',
                                border: '#2E7D32',
                                highlight: {
                                    background: '#81C784',
                                    border: '#2E7D32'
                                }
                            }
                        },
                        counterparty: {
                            color: {
                                background: '#2196F3',
                                border: '#1565C0',
                                highlight: {
                                    background: '#64B5F6',
                                    border: '#1565C0'
                                }
                            }
                        },
                        ip: {
                            color: {
                                background: '#FF5722',
                                border: '#D84315',
                                highlight: {
                                    background: '#FF8A65',
                                    border: '#D84315'
                                }
                            }
                        },
                        mac: {
                            color: {
                                background: '#9C27B0',
                                border: '#6A1B9A',
                                highlight: {
                                    background: '#BA68C8',
                                    border: '#6A1B9A'
                                }
                            }
                        },
                        account: {
                            color: {
                                background: '#FFEB3B',
                                border: '#F9A825',
                                highlight: {
                                    background: '#FFF176',
                                    border: '#F9A825'
                                }
                            }
                        },
                        card: {
                            color: {
                                background: '#00BCD4',
                                border: '#00838F',
                                highlight: {
                                    background: '#4DD0E1',
                                    border: '#00838F'
                                }
                            }
                        }
                    }
                };

                // 创建网络图
                var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);

                // 重置视图
                function resetView() {
                    network.fit({
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });

                    // 重置所有节点的透明度
                    var allNodes = nodes.get();
                    for (var i = 0; i < allNodes.length; i++) {
                        delete allNodes[i].color;
                    }
                    nodes.update(allNodes);
                }

                // 搜索功能
                var searchInput = document.getElementById('search-input');
                var searchResults = document.getElementById('search-results');

                searchInput.addEventListener('input', function() {
                    var query = this.value.toLowerCase();
                    if (query.length < 2) {
                        searchResults.style.display = 'none';
                        return;
                    }

                    var matches = [];
                    var allNodes = nodes.get();

                    for (var i = 0; i < allNodes.length; i++) {
                        var node = allNodes[i];
                        if (node.label.toLowerCase().indexOf(query) !== -1) {
                            matches.push(node);
                        }
                    }

                    // 显示搜索结果
                    searchResults.innerHTML = '';
                    if (matches.length > 0) {
                        for (var i = 0; i < Math.min(matches.length, 10); i++) {
                            var div = document.createElement('div');
                            div.className = 'search-result';
                            div.textContent = matches[i].label;
                            div.dataset.id = matches[i].id;
                            div.addEventListener('click', function() {
                                var nodeId = this.dataset.id;
                                network.focus(nodeId, {
                                    scale: 1.5,
                                    animation: {
                                        duration: 1000,
                                        easingFunction: 'easeInOutQuad'
                                    }
                                });
                                network.selectNodes([nodeId]);
                                searchResults.style.display = 'none';
                                searchInput.value = '';
                            });
                            searchResults.appendChild(div);
                        }
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.style.display = 'none';
                    }
                });

                // 点击其他地方时隐藏搜索结果
                document.addEventListener('click', function(e) {
                    if (e.target !== searchInput && e.target !== searchResults) {
                        searchResults.style.display = 'none';
                    }
                });

                // 切换物理引擎
                var physicsEnabled = true;
                function togglePhysics() {
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({ physics: { enabled: physicsEnabled } });
                }

                // 双击节点时聚焦并显示相邻节点
                network.on("doubleClick", function(params) {
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var connectedNodes = network.getConnectedNodes(nodeId);
                        var allNodes = nodes.get();

                        // 重置所有节点的透明度
                        for (var i = 0; i < allNodes.length; i++) {
                            allNodes[i].color = {
                                opacity: 0.3
                            };
                        }

                        // 高亮选中节点和相邻节点
                        var selectedNode = nodes.get(nodeId);
                        selectedNode.color = {
                            opacity: 1.0
                        };

                        for (var i = 0; i < connectedNodes.length; i++) {
                            var connectedNode = nodes.get(connectedNodes[i]);
                            connectedNode.color = {
                                opacity: 1.0
                            };
                        }

                        nodes.update(allNodes);
                    } else {
                        // 如果双击空白处，重置所有节点的透明度
                        var allNodes = nodes.get();
                        for (var i = 0; i < allNodes.length; i++) {
                            delete allNodes[i].color;
                        }
                        nodes.update(allNodes);
                    }
                });
            </script>
        </body>
        </html>
        """
        # Prepare nodes and edges data for the visualization
        nodes_data = []
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            # Set node color and group based on type
            if node_type == 'customer':
                color = '#4CAF50'  # Green for customers
                group = 'customer'
            elif node_type == 'counterparty':
                color = '#2196F3'  # Blue for counterparties
                group = 'counterparty'
            elif node_type == 'ip':
                color = '#FF5722'  # Orange for IP addresses
                group = 'ip'
            elif node_type == 'mac':
                color = '#9C27B0'  # Purple for MAC addresses
                group = 'mac'
            elif node_type == 'account':
                color = '#FFEB3B'  # Yellow for accounts
                group = 'account'
            elif node_type == 'card':
                color = '#00BCD4'  # Cyan for cards
                group = 'card'
            else:
                color = '#9E9E9E'  # Grey for other nodes
                group = 'other'

            # Create a label that's not too long
            if ':' in node:
                prefix, value = node.split(':', 1)
                if len(value) > 15:
                    label = f"{prefix}:{value[:12]}..."
                else:
                    label = node
            else:
                label = node

            nodes_data.append({
                'id': node,
                'label': label,
                'title': attrs.get('title', node),
                'value': attrs.get('size', 5),  # Use value for size in vis.js
                'color': color,
                'group': group
            })

        edges_data = []
        for source, target, attrs in G.edges(data=True):
            width = min(attrs.get('weight', 1), 10)  # Cap width at 10

            # Set edge color based on relationship type
            if 'relationship' in attrs:
                if attrs['relationship'] == '共用IP':
                    color = '#FF5722'  # Orange for shared IP
                elif attrs['relationship'] == '共用MAC':
                    color = '#9C27B0'  # Purple for shared MAC
                elif attrs['relationship'] == '共用账号':
                    color = '#FFEB3B'  # Yellow for shared account
                elif attrs['relationship'] == '共用卡号':
                    color = '#00BCD4'  # Cyan for shared card
                elif attrs['relationship'] == '使用IP':
                    color = '#FF5722'  # Orange for IP usage
                elif attrs['relationship'] == '使用MAC':
                    color = '#9C27B0'  # Purple for MAC usage
                elif attrs['relationship'] == '拥有账号':
                    color = '#FFEB3B'  # Yellow for account ownership
                elif attrs['relationship'] == '拥有卡':
                    color = '#00BCD4'  # Cyan for card ownership
                elif attrs['relationship'] == '交易':
                    color = '#2196F3'  # Blue for transactions
                else:
                    color = '#9E9E9E'  # Grey for other relationships
            else:
                color = '#9E9E9E'  # Grey for undefined relationships

            edges_data.append({
                'from': source,
                'to': target,
                'title': attrs.get('title', ''),
                'width': width,
                'color': color
            })

        # Replace placeholders in the template
        html_content = html_template.replace('NODES_DATA', json.dumps(nodes_data))
        html_content = html_content.replace('EDGES_DATA', json.dumps(edges_data))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)

        # Write the HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Interactive knowledge graph visualization saved to {output_file}")

        html_generate_state = True
#        # Open the HTML file in the default browser
#        try:
#            webbrowser.open('file://' + os.path.abspath(output_file))
#        except:
#            print("Could not automatically open the visualization in a browser.")
    except Exception as e:
        print(f"Error creating knowledge graph: {e}")

    static_output = None
    try:
        print(f"Error creating interactive visualization: {e}")
        print("Falling back to static visualization...")

        # Fallback to static visualization with matplotlib
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, seed=42)  # For reproducibility

        # Draw nodes with different colors based on type
        customer_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'customer']
        counterparty_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'counterparty']
        ip_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'ip']
        mac_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'mac']
        account_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'account']
        card_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'card']
        other_nodes = [n for n in G.nodes() if n not in customer_nodes and n not in counterparty_nodes 
                        and n not in ip_nodes and n not in mac_nodes and n not in account_nodes and n not in card_nodes]

        # Draw edges with different colors based on relationship
        edge_colors = []
        for u, v, attrs in G.edges(data=True):
            if 'relationship' in attrs:
                if '共用IP' in attrs['relationship'] or '使用IP' in attrs['relationship']:
                    edge_colors.append('orange')
                elif '共用MAC' in attrs['relationship'] or '使用MAC' in attrs['relationship']:
                    edge_colors.append('purple')
                elif '共用账号' in attrs['relationship'] or '拥有账号' in attrs['relationship']:
                    edge_colors.append('yellow')
                elif '共用卡号' in attrs['relationship'] or '拥有卡' in attrs['relationship']:
                    edge_colors.append('cyan')
                elif '交易' in attrs['relationship']:
                    edge_colors.append('blue')
                else:
                    edge_colors.append('grey')
            else:
                edge_colors.append('grey')

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, nodelist=customer_nodes, node_color='green', node_size=700, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=counterparty_nodes, node_color='blue', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=ip_nodes, node_color='orange', node_size=300, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=mac_nodes, node_color='purple', node_size=300, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=account_nodes, node_color='yellow', node_size=300, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=card_nodes, node_color='cyan', node_size=300, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='grey', node_size=200, alpha=0.8)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        plt.axis('off')
        plt.tight_layout()

        # Save as PNG
        static_output = output_file.replace('.html', '.png')
        plt.savefig(static_output, dpi=300)
        print(f"Static knowledge graph visualization saved to {static_output}")

        # Also save as GraphML for potential use with other tools
        graphml_output = output_file.replace('.html', '.graphml')
        nx.write_graphml(G, graphml_output)
        print(f"Graph data saved in Graphml format: {graphml_output}")

        png_generate_state = True
    except Exception as e:
        print(f"Error creating static visualization: {e}")

    return output_file, html_generate_state, static_output, png_generate_state

# 使用示例
if __name__ == "__main__":
    import argparse
    import os

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='处理交易数据并生成知识图谱')
    parser.add_argument('--input', type=str, default='example_data_gang_profits.csv', help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录路径')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置输出文件路径
    json_file = os.path.join(args.output_dir, "compressed_data.json")
    graph_file = os.path.join(args.output_dir, "transaction_knowledge_graph.html")

    # 1. 压缩数据
    print(f"正在处理CSV文件: {args.input}")
    data = compress_data(args.input, json_file)
    print(f"数据压缩完成，已保存至: {json_file}")

    # 2. 生成知识图谱
    print("正在生成知识图谱...")
    G = create_knowledge_graph(data, graph_file)
    print(f"知识图谱生成完成，已保存至: {graph_file}")
    print("处理完成!")
