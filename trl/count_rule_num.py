import json

def count_rule_num(file_path):
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    print(f"File path: {file_path}, rule count: {len(data)}")




if __name__ == "__main__":
    count_rule_num("data/sse_trading_rules.json")
    count_rule_num("data/szse_bond_trading_rules.json")
    count_rule_num("data/szse_fund_trading_and_redemption.json")