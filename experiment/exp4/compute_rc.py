import copy
import json
import re
import os
import nltk
from testcase.generate_testcase import judge_op, is_num_key, is_price_key, is_time_key
import argparse




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def edit_distance(s1, s2):
    """
    Measure the similarity between two strings by computing their edit distance. The edit distance is the minimum number of operations required to transform one string into another, including insertions, deletions, and substitutions.
    """
    return nltk.edit_distance(s1, s2)
    # len1, len2 = len(s1), len(s2)
    # dp = [[0] * (len2+1) for _ in range(len1+1)]
    # for i in range(len1+1):
    #     dp[i][0] = i
    # for j in range(len2+1):
    #     dp[0][j] = j
    # for i in range(1, len1+1):
    #     for j in range(1, len2+1):
    #         if s1[i-1] == s2[j-1]:
    #             dp[i][j] = dp[i-1][j-1]
    #         else:
    #             dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    # return dp[len1][len2]

def str_same(s1, s2, threshold):
    # s1 is the predicted value, s2 is the ground-truth value
    distance = edit_distance(s1, s2)
    # score = 1 - min(1, distance / min(len(s1), len(s2))) if min(len(s1), len(s2)) > 0 else 0
    score = 1 - distance / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0
    if score >= threshold:
        return True
    else:
        return False


def judge_same(s1, s2, metric_rc):
    s1, s2 = str(s1), str(s2)
    return str_same(s1, s2, metric_rc)


def is_number(s):
    try:
        s = float(s)
        return True
    except ValueError:
        return False



def compute_bsc_v2(testcases, scenarios, metric_rc):
    """
    For a given scenario and a given testcase, this function determines whether each variable of the scenario is mentioned in the testcase (i.e. has the same value),
    then the coverage of a scenario = number of mentioned variables / total number of variables, and the highest coverage is taken.
    The overall coverage = the mean of all scenario coverages.
    """
    # Preprocess the scenarios
    new_scenarios = []  # example: new_scenarios[i]['交易市场'] (market key) = '深圳证券交易所' (a market-name value)
    scenarios_variables = []  # scenario_variables[i]['交易市场'] = 0 means the element is not covered, 1 means covered
    max_cover_varnum = [0] * len(scenarios)  # the maximum number of covered variables for each test scenario
    cover_rate = 0

    for scenario in scenarios:
        s = {}
        variables = {}
        scs = scenario.split(";")
        for sc in scs:
            if "时间" not in sc:
                ss = sc.split(":")
                s[ss[0]] = ss[1]
                variables[ss[0]] = 0
            else:
                ss = sc.split(":")
                s[ss[0]] = ":".join(ss[1:])
                variables[ss[0]] = 0
        new_scenarios.append(s)
        scenarios_variables.append(variables)
    scenarios = new_scenarios


    for scenario_index, scenario in enumerate(scenarios):
        scenario_variables_total = copy.deepcopy(scenarios_variables[scenario_index])
        for testcase_index, testcase in enumerate(testcases):
            scenario_variables = copy.deepcopy(scenarios_variables[scenario_index])
            
            # Compute how much this testcase covers in the scenario
            for testcase_key, testcase_value in testcase.items():
                # Skip irrelevant testcase keys
                if testcase_key in ['rule', '测试关注点', 'testid']:
                    continue
                if isinstance(testcase_value, list) or isinstance(testcase_value, dict):
                    # If it is a list or dict, skip directly
                    continue
                testcase_value = str(testcase_value).replace(" ", "").strip()
                for scenario_key, scenario_value in scenario.items():
                    # Both are time variables
                    if not judge_same(scenario_key, testcase_key, metric_rc):
                        continue
                    if is_time_key(testcase_key) and is_time_key(scenario_key):
                        if ":" not in testcase_value and ":" not in scenario_value:  # cases like time is "上市首日" (first listing day), handled as enum variables
                            for s_value in scenario_value.split(","):
                                if judge_same(testcase_value, s_value, metric_rc):
                                    scenario_variables[scenario_key] = 1
                                    break
                                # else: if the times conflict, do not mark the corresponding variable as covered, continue comparing
                            continue
                        elif ":" not in testcase_value or ":" not in scenario_value:
                            continue
                        # else: two time variables, convert to the same format for comparison
                        # testcase_value example: 00:00:00-09:30:00 or (或) 11:30:00-13:00:00 or (或) 14:57:00-24:00:00
                        # scenario_value example: 非(not)9:15至(to)11:30,13:00至(to)15:30
                        vs = [t.strip() for t in testcase_value.split("或")]
                        t_value = "{"
                        for v in vs:
                            t_value += f"[{v}],"
                        t_value = t_value[:-1] + "}"
                        fei = False
                        if "非" == scenario_value[0]:
                            fei = True
                            scenario_value = scenario_value[1:]
                        vs = scenario_value.split(",")
                        time = []
                        for v in vs:
                            if "至" in v or "-" in v:
                                t1 = v.split("至")[0] if "至" in v else v.split("-")[0]
                                t2 = v.split("至")[1] if "至" in v else v.split("-")[1]
                                if len(t1) == 4:
                                    t1 = "0" + t1 + ":00"
                                elif len(t1) == 5:
                                    t1 = t1 + ":00"
                                elif len(t1) == 7:
                                    t1 = "0" + t1
                                if len(t2) == 4:
                                    t2 = "0" + t2 + ":00"
                                elif len(t2) == 5:
                                    t2 = t2 + ":00"
                                elif len(t2) == 7:
                                    t2 = "0" + t2
                                time.append(t1)
                                time.append(t2)
                            elif "前" in v or "后" in v:
                                t = v[:-1]
                                if len(t) == 4:
                                    t = "0" + t + ":00"
                                elif len(t) == 5:
                                    t = t + ":00"
                                elif len(t) == 7:
                                    t = "0" + t
                                if "前" in v:
                                    time.append("00:00:00")
                                    time.append(t)
                                else:
                                    time.append(t)
                                    time.append("24:00:00")
                        if fei:
                            if time[0] == "00:00:00":
                                del time[0]
                            else:
                                time.insert(0, "00:00:00")
                            if time[-1] == "24:00:00":
                                del time[-1]
                            else:
                                time.append("24:00:00")
                        s_value = "{"
                        for i in range(0, len(time), 2):
                            s_value += f"[{time[i]}-{time[i+1]}],"
                        s_value = s_value[:-1] + "}"
                        if s_value == t_value:
                            scenario_variables[scenario_key] = 1

                    # Both are quantity variables
                    elif is_num_key(testcase_key) and is_num_key(scenario_key):
                        
                        if "一次性" in scenario_value and "(余额" in testcase_value:
                            scenario_variables[scenario_key] = 1
                            continue
                        elif "一次性" in scenario_value or "(余额)" in testcase_value:
                            continue
                        # else: regular numeric constraint or enum constraint
                        nums = [t.strip() for t in testcase_value.split("或")]
                        fei = False
                        if "非" == scenario_value[0]:
                            scenario_value = scenario_value[1:]
                            fei = True
                        fulfill_all = True  # assume here that all constraints are satisfied
                        for sv in scenario_value.split(","):
                            fulfill = False
                            for num in nums:
                                if is_number(num):
                                    num = int(num) if "." not in num else int(float(num))
                                else:  # enum constraint
                                    if judge_same(num, sv, metric_rc):
                                        fulfill = True
                                        break
                                    else:
                                        continue
                                if "整数倍" in sv:
                                    if len(re.findall(r"\d+", sv)) > 0:
                                        value = int(re.findall(r"\d+", sv)[0])  # integer multiple of value
                                        if num % value == 0 and not fei or num % value != 0 and fei:
                                            # condition satisfied
                                            fulfill = True
                                            break
                                    else:
                                        continue
                                op = judge_op(sv)
                                all_v = re.findall(f"\d+", sv)
                                if len(all_v)>0:
                                    value = float(all_v[0])  # op value
                                else:  # the price in the scenario is an enum variable (but price is not)
                                    continue
                                if "万" in sv:
                                    value = value * 10000
                                if "亿" in sv:
                                    value = value * 100000000
                                constraint_fulfill = op == ">=" and num >= value or op == "<=" and num <= value or op == ">" and num > value or op == "<" and num < value or op == "==" and num == value or op == "!=" and num != value
                                fulfill = constraint_fulfill and not fei or not constraint_fulfill and fei
                                if fulfill:
                                    break
                            # For a positive case, fulfill_all is required; for a negative case, a single fulfill counts as success
                            if fei and fulfill:
                                fulfill_all = True
                                break
                            if not fulfill:
                                fulfill_all = False
                                break
                        if fulfill_all:
                            scenario_variables[scenario_key] = 1

                    # Both are price variables
                    elif is_price_key(testcase_key) and is_price_key(scenario_key):
                        prices = [t.strip() for t in testcase_value.split("或")]
                        fei = False
                        if "非" == scenario_value[0]:
                            scenario_value = scenario_value[1:]
                            fei = True
                        fulfill_all = True  # assume here that all constraints are satisfied
                        for sv in scenario_value.split(","):
                            fulfill = False
                            for price in prices:
                                if is_number(price):
                                    price = float(price)
                                else:
                                    if judge_same(price, sv, metric_rc):
                                        fulfill = True
                                        break
                                    else:
                                        continue
                                op = judge_op(sv)
                                if op == "":
                                    op = "=="
                                all_v = re.findall(r"\d+\.\d+|\d+", sv)
                                if len(all_v)>0:
                                    value = float(all_v[0])  # op value
                                else:  # the price in the scenario is an enum variable (but price is not)
                                    continue
                                if "万" in sv:
                                    value = value * 10000
                                if "亿" in sv:
                                    value = value * 100000000
                                constraint_fulfill = op == ">=" and price >= value or op == "<=" and price <= value or op == ">" and price > value or op == "<" and price < value or op == "==" and price == value or op == "!=" and price != value
                                fulfill = constraint_fulfill and not fei or not constraint_fulfill and fei
                                if fulfill:
                                    break
                            if fei and fulfill:
                                fulfill_all = True
                                break
                            if not fulfill:
                                fulfill_all = False
                                break
                        if fulfill_all:
                            scenario_variables[scenario_key] = 1

                    # Both are enum variables
                    elif not is_time_key(testcase_key) and not is_time_key(testcase_key) and not is_num_key(testcase_key) and not is_num_key(scenario_key) and not is_price_key(testcase_key) and not is_price_key(scenario_key):
                        
                        # For an enum variable in the scenario, if a string with a similar value exists in the testcase, count it as covered; otherwise not
                        for s_value in scenario_value.split(","):
                            if judge_same(testcase_value, s_value, metric_rc):
                                scenario_variables[scenario_key] = 1
                                break
            
            if "testid" in testcase:
                print(f"## 测试场景\"{scenario_index+1}\", 测试用例\"{testcase['testid']}\", 覆盖变量数目为{sum(scenario_variables.values())}, 未覆盖的变量包括{[key for key in scenario_variables.keys() if scenario_variables[key] == 0]}\n")
            else:
                print(f"## 测试场景\"{scenario_index+1}\", 测试用例\"{testcase_index}\", 覆盖变量数目为{sum(scenario_variables.values())}, 未覆盖的变量包括{[key for key in scenario_variables.keys() if scenario_variables[key] == 0]}\n")
            
            for key in scenario_variables_total.keys():
                if scenario_variables[key] == 1:
                    scenario_variables_total[key] = 1
        
        max_cover_varnum[scenario_index] = sum(scenario_variables_total.values())
        if len(scenario_variables_total.keys()) > max_cover_varnum[scenario_index]:
            print(f"### 测试场景\"{scenario_index+1}\", 覆盖变量的最大数目为{max_cover_varnum[scenario_index]}, 整体未覆盖的变量包括{[key for key in scenario_variables_total.keys() if scenario_variables_total[key] == 0]}\n\n")
        else:
            print(f"### 测试场景\"{scenario_index+1}\", 覆盖变量的最大数目为{max_cover_varnum[scenario_index]}, 所有变量全部覆盖\n\n")
    
    max_cover_rate = [max_cover_varn / len(scenarios[i]) for i, max_cover_varn in enumerate(max_cover_varnum)]
    cover_rate += sum(max_cover_rate) / len(max_cover_rate)
    return cover_rate


def compute_bsc_v1(testcases, scenarios, f, metric_rc):
    """
    For a given test scenario and a given testcase, this function checks whether each element in the testcase appears in the scenario.
    If it appears and is similar, it is considered correct; if not similar, it is considered a conflict; if it does not appear, it is assumed to appear by default.
    This makes the resulting accuracy skew high.
    """
    # Preprocess the scenarios
    if_cover = [0] * len(scenarios)
    new_scenarios = []
    for scenario in scenarios:
        s = {}
        scs = scenario.split(";")
        for sc in scs:
            if "时间" not in sc:
                ss = sc.split(":")
                s[ss[0]] = ss[1]
            else:
                ss = sc.split(":")
                s[ss[0]] = ":".join(ss[1:])
        new_scenarios.append(s)
    scenarios = new_scenarios

    for testcase in testcases:
        for t in testcase:
            for iis, s in enumerate(scenarios):
                s_keys = list(s.keys())
                t_keys = list(t.keys())
                conflict = False
                find_time, find_num, find_price = False, False, False  # number of matches
                # Count the time, quantity, and price keys in s
                s_time_keys = []
                for s_key in s_keys:
                    if "时间" in s_key:
                        s_time_keys.append(s_key)
                
                s_num_keys = []
                for s_key in s_keys:
                    if "数量" in s_key:
                        s_num_keys.append(s_key)
                
                s_price_keys = []
                for s_key in s_keys:
                    if "价格" in s_key or "金额" in s_key:
                        s_price_keys.append(s_key)

                for t_key in t_keys:
                    if t_key in ["rule", "测试关注点", "testid"]:
                        continue
                    if t_key == "结果":
                        # must be identical
                        if t[t_key] != s[t_key]:
                            conflict = True
                            break
                    elif not is_time_key(t_key) and not is_num_key(t_key) and not is_price_key(t_key):
                        # enum constraint
                        # if an identical value is found, count it as the same
                        find = False
                        for s_key in s_keys:
                            for s_value in s[s_key].split(","):
                                if judge_same(t[t_key], s_value, metric_rc):
                                    find = True
                                    break
                            if find:
                                break
                        if find:
                            continue
                        # no identical value was found, check whether there is a conflict
                        if t_key not in s_keys:
                            continue
                        else:
                            conflict = True
                            break
                    elif is_time_key(t_key):
                        if len(s_time_keys) == 0:
                            conflict = True
                            break
                        find = False
                        for s_key in s_time_keys:
                            t_value = t[t_key]
                            s_value = s[s_key]
                            if ":" not in t_value and ":" not in s_value:  # cases like time is "上市首日" (first listing day)
                                if judge_same(t_value, s_value, metric_rc):
                                    find = True
                                    break
                                else:
                                    continue
                            elif ":" not in t_value or ":" not in s_value:
                                continue
                            # t_value example: 00:00:00-09:30:00 or (或) 11:30:00-13:00:00 or (或) 14:57:00-24:00:00
                            # s_value example: 非(not)9:15至(to)11:30,13:00至(to)15:30
                            # Convert the s_value and t_value formats
                            vs = [t.strip() for t in t_value.split("或")]
                            t_value = "{"
                            for v in vs:
                                t_value += f"[{v}],"
                            t_value = t_value[:-1] + "}"
                            fei = False
                            if "非" in s_value:
                                fei = True
                                s_value = s_value[1:]
                            vs = s_value.split(",")
                            time = []
                            for v in vs:
                                if "至" in v or "-" in v:
                                    t1 = v.split("至")[0] if "至" in v else v.split("-")[0]
                                    t2 = v.split("至")[1] if "至" in v else v.split("-")[1]
                                    if len(t1) == 4:
                                        t1 = "0" + t1 + ":00"
                                    elif len(t1) == 5:
                                        t1 = t1 + ":00"
                                    elif len(t1) == 7:
                                        t1 = "0" + t1
                                    if len(t2) == 4:
                                        t2 = "0" + t2 + ":00"
                                    elif len(t2) == 5:
                                        t2 = t2 + ":00"
                                    elif len(t2) == 7:
                                        t2 = "0" + t2
                                    time.append(t1)
                                    time.append(t2)
                                elif "前" in v or "后" in v:
                                    t = v[:-1]
                                    if len(t) == 4:
                                        t = "0" + t + ":00"
                                    elif len(t) == 5:
                                        t = t + ":00"
                                    elif len(t) == 7:
                                        t = "0" + t
                                    if "前" in v:
                                        time.append("00:00:00")
                                        time.append(t)
                                    else:
                                        time.append(t)
                                        time.append("24:00:00")
                            if fei:
                                if time[0] == "00:00:00":
                                    del time[0]
                                else:
                                    time.insert(0, "00:00:00")
                                if time[-1] == "24:00:00":
                                    del time[-1]
                                else:
                                    time.append("24:00:00")
                            s_value = "{"
                            for i in range(0, len(time), 2):
                                s_value += f"[{time[i]}-{time[i+1]}],"
                            s_value = s_value[:-1] + "}"
                            if s_value == t_value:
                                find = True
                                break
                        if find:
                            find_time = True

                    elif is_num_key(t_key):
                        if len(s_num_keys) == 0:
                            conflict = True
                            break
                        find = False
                        for s_key in s_num_keys:
                            t_value = t[t_key]
                            s_value = s[s_key]
                            if "一次性" in s_value and "(余额)" in t_value:
                                find = True
                                break
                            if "一次性" in s_value or "(余额)" in t_value:
                                continue
                            nums = [t.strip() for t in t_value.split("或")]
                            fei = False
                            if "非" in s_value:
                                s_value = s_value[1:]
                                fei = True
                            fulfill_all = True  # assume here that all constraints are satisfied
                            for sv in s_value.split(","):
                                fulfill = False
                                for num in nums:
                                    if is_number(num):
                                        num = int(num)
                                    else:
                                        if judge_same(num, sv, metric_rc):
                                            fulfill = True
                                            break
                                        else:
                                            continue
                                    if "整数倍" in sv:
                                        value = int(re.findall(r"\d+", sv)[0])  # integer multiple of value
                                        if num % value == 0 and not fei or num % value != 0 and fei:
                                            # condition satisfied
                                            fulfill = True
                                            break
                                    op = judge_op(sv)
                                    value = int(re.findall(r"\d+", sv)[0])  # op value
                                    constraint_fulfill = op == ">=" and num >= value or op == "<=" and num <= value or op == ">" and num > value or op == ">" and num > value or op == "==" and num == value or op == "!=" and num != value
                                    fulfill = constraint_fulfill and not fei or not constraint_fulfill and fei
                                    if fulfill:
                                        break
                                if not fulfill:
                                    fulfill_all = False
                                    break
                            if fulfill_all:
                                find = True
                                break
                        if find:
                            find_num = True

                    else:  # "价格"/"金额" (price/amount) in t_key
                        if len(s_price_keys) == 0:
                            conflict = True
                            break
                        find = False
                        for s_key in s_price_keys:
                            t_value = t[t_key]
                            s_value = s[s_key]
                            prices = [t.strip() for t in t_value.split("或")]
                            fei = False
                            if "非" in s_value:
                                s_value = s_value[1:]
                                fei = True
                            fulfill_all = True  # assume here that all constraints are satisfied
                            for sv in s_value.split(","):
                                fulfill = False
                                for price in prices:
                                    if is_number(price):
                                        price = float(price)
                                    else:
                                        if judge_same(price, sv, metric_rc):
                                            fulfill = True
                                            break
                                        else:
                                            continue
                                    op = judge_op(sv)
                                    value = float(re.findall(r"\d+", sv)[0])  # op value
                                    constraint_fulfill = op == ">=" and price >= value or op == "<=" and price <= value or op == ">" and price > value or op == ">" and price > value or op == "==" and price == value or op == "!=" and price != value
                                    fulfill = constraint_fulfill and not fei or not constraint_fulfill and fei
                                    if fulfill:
                                        break
                                if not fulfill:
                                    fulfill_all = False
                                    break
                            if fulfill_all:
                                find = True
                                break
                        if find:
                            find_price = True
                if (len(s_time_keys) == 0 or len(s_time_keys) > 0 and find_time) or (len(s_num_keys) == 0 or len(s_num_keys) > 0 and find_num) or (len(s_price_keys) == 0 or len(s_price_keys) > 0 and find_price):
                    ...
                else:
                    conflict = True
                if not conflict:
                    if_cover[iis] = 1

    for i, cover in enumerate(if_cover):
        if cover == 0:
            f.write(str(i+1))
            f.write("\n")
    return float(sum(if_cover)) / float(len(if_cover))
metric = json.load(open("../exp1/log/o_config.json", 'r', encoding='utf-8'))['exp1']

def compute_bsc_v3(testcases, scenarios):
    cover = []
    for scenario in scenarios:
        cover.append([False for _ in range(len(scenario))])
        for testcase in testcases:
            for key, value in testcase.items():
                if isinstance(value, list) or isinstance(value, dict):
                    continue
                value = str(value).strip()
                i = 0
                while key in scenario[i:]:
                    j = scenario.index(key, i)
                    i = j + 1
                    cover[-1][j:j + len(key)] = [True] * len(key)
                i = 0
                while value in scenario[i:]:
                    if value == "":
                        break
                    j = scenario.index(value, i)
                    i = j + 1
                    cover[-1][j:j + len(value)] = [True] * len(value)
    cover_rate = []
    for i, c in enumerate(cover):
        cover_rate.append(sum(c) / len(c))
    acc = sum(cover_rate) / len(cover_rate)
    return acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--method", default=None, type=str)
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method

    result = {}
    for file in sorted(os.listdir("requirement")):
        if dataset is None or dataset not in file:
            continue
        testcase_file = f"result/{method}/{dataset}.json"
        scenario_file = f"requirement/{file}"
        testcases = json.load(open(testcase_file, "r", encoding="utf-8"))
        testcases = [t['testcase'] for t in testcases]
        testcases = [t for testcase_ in testcases for t in testcase_]
        scenarios = open(scenario_file, "r", encoding="utf-8").read().strip().split("\n")
        bsc = compute_bsc_v2(testcases, scenarios, metric['only_rc '])
        print(f"{method}数据集{file.split('_')[0]}的需求覆盖率为{round(bsc, 4)}")
        result[dataset] = bsc
    json.dump(result, open(f"log/{method}_rc_{dataset}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)