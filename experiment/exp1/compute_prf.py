import json
from testcase.generate_testcase import judge_op, is_num_key, is_price_key, is_time_key
import copy
import os
from nltk import edit_distance
import re
import argparse
from tqdm import tqdm


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def str_same(s1, s2, threshold):
    """
    Whether the similarity between s1 and s2 is greater than threshold.
    """
    if len(s1) == 0 or len(s2) == 0:
        return False
    return 1 - edit_distance(s1, s2) / max(len(s1), len(s2)) > threshold

def judge_same(t1, t2, threshold, strict=False):
    """
    Whether a threshold fraction of elements in t1 and t2 are similar, where each element's value similarity is greater than threshold.
    """
    t1_keys, t1_values, t2_keys, t2_values = [], {}, [], {}
    for k, v in t1.items():
        if k == "testid" or k == "rule" or k == "id":
            continue
        # strip the trailing digits from the key
        k = re.sub(r'\d+$', '', k)
        if k in t1_keys:
            t1_values[k].append(v)
        else:
            t1_keys.append(k)
            t1_values[k] = [v]
    for k, v in t2.items():
        if k == "testid" or k == "rule" or k == "id":
            continue
        k = re.sub(r'\d+$', '', k)
        if k in t2_keys:
            t2_values[k].append(v)
        else:
            t2_keys.append(k)
            t2_values[k] = [v]
    
    if not strict:
        # fuzzy matching method
        t1_like = 0
        for k1 in t1_keys:
            v1 = t1_values[k1]
            for k2 in t2_keys:
                v2 = t2_values[k2]
                if not str_same(k1, k2, threshold):
                    continue
                v1_like = 0
                for vi in v1:
                    for vj in v2:
                        if str_same(vi, vj, threshold):
                            v1_like += 1
                            break
                v1_like /= len(v1)
                if v1_like > threshold:
                    t1_like += 1
                    break
        return t1_like / len(t1_keys) > threshold
    else:
        t1_keys, t2_keys = sorted(t1_keys), sorted(t2_keys)
        if t1_keys != t2_keys:
            return False
        for k in t1_keys:
            if sorted(t1_values[k]) != sorted(t2_values[k]):
                return False
        return True



def eval_testcase(ours_testcases, label_testcases, metric_precision, metric_recall):
    """
    Compute the accuracy between our generated testcases and the label testcases.
    """
    for t in ours_testcases:
        if not isinstance(t, dict):
            bug_file = open("bug.log", 'a', encoding='utf-8')
            bug_file.write(json.dumps(t, ensure_ascii=False, indent=4))
            bug_file.close()
            exit(-1)
        new_t = {}
        for k, v in t.items():
            if v == None:
                continue
            elif not isinstance(v, str):
                new_t[k] = json.dumps(v, ensure_ascii=False)
            else:
                new_t[k] = v
        t.clear()
        t.update(new_t)
    for t in label_testcases:
        if not isinstance(t, dict):
            bug_file = open("bug.log", 'a', encoding='utf-8')
            bug_file.write(json.dumps(t, ensure_ascii=False, indent=4))
            bug_file.close()
            exit(-1)
        new_t = {}
        for k, v in t.items():
            if v == None:
                continue
            elif not isinstance(v, str):
                new_t[k] = json.dumps(v, ensure_ascii=False)
            else:
                new_t[k] = v
        t.clear()
        t.update(new_t)
    find = [False for _ in range(len(label_testcases))]
    find_ = [False for _ in range(len(ours_testcases))]
    for i, testcase in enumerate(tqdm(label_testcases)):
        for j, t in enumerate(ours_testcases):
            if judge_same(testcase, t, metric_recall):
                find[i] = True
                break

    for j, t in enumerate(tqdm(ours_testcases)):
        for i, testcase in enumerate(label_testcases):
            if judge_same(testcase, t, metric_precision):
                find_[j] = True
                break

    recall = sum(find) / len(label_testcases)
    precision = sum(find_) / len(ours_testcases)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def compute_bsc_v1(testcases, scenarios, f):
    """
    For a given test scenario and a given testcase, this function checks, for each element in the testcase, whether it appears in the scenario.
    If it appears and is similar, it is treated as correct; if not similar, as a conflict; if it does not appear, it defaults to appearing.
    As a result the accuracy tends to be on the high side.
    """
    # preprocess scenarios
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
                # count the time / quantity / price keys in s
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
                        # if an identical value is found, treat them as identical
                        find = False
                        for s_key in s_keys:
                            for s_value in s[s_key].split(","):
                                if judge_same(t[t_key], s_value):
                                    find = True
                                    break
                            if find:
                                break
                        if find:
                            continue
                        # no identical value found; check whether they conflict
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
                            if ":" not in t_value and ":" not in s_value:  # time is "上市首日" (first listing day)
                                if judge_same(t_value, s_value):
                                    find = True
                                    break
                                else:
                                    continue
                            elif ":" not in t_value or ":" not in s_value:
                                continue
                            # t_value example: 00:00:00-09:30:00 or (或) 11:30:00-13:00:00 or (或) 14:57:00-24:00:00
                            # s_value example: 非(not)9:15至(to)11:30,13:00至(to)15:30
                            # convert the s_value and t_value formats
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
                            fulfill_all = True  # here we assume all constraints are satisfied
                            for sv in s_value.split(","):
                                fulfill = False
                                for num in nums:
                                    if is_number(num):
                                        num = int(num)
                                    else:
                                        if judge_same(num,sv):
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
                            fulfill_all = True  # here we assume all constraints are satisfied
                            for sv in s_value.split(","):
                                fulfill = False
                                for price in prices:
                                    if is_number(price):
                                        price = float(price)
                                    else:
                                        if judge_same(price, sv):
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
metric = json.load(open("log/o_config.json", 'r', encoding='utf-8'))['exp1']

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

def compute_bsc_v2(testcases, scenarios):
    """
    For a given scenario and a given testcase, this function determines whether each variable of the scenario is mentioned in the testcase (same value).
    Then the scenario coverage = number of mentioned variables / total number of variables, taking the highest coverage.
    The overall coverage = the mean of all scenario coverages.
    """
    # preprocess scenarios
    new_scenarios = []  # example: new_scenarios[i]['交易市场'] (market key) = '深圳证券交易所' (a market-name value)
    scenarios_variables = []  # scenario_variables[i]['交易市场'] = 0 means the element is not covered, 1 means covered
    max_cover_varnum = [0] * len(scenarios)  # the maximum number of covered variables per test scenario
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
            
            # compute how much this testcase covers in the scenario
            for testcase_key, testcase_value in testcase.items():
                # skip irrelevant testcase keys
                if testcase_key in ['rule', '测试关注点', 'testid']:
                    continue
                if isinstance(testcase_value, list) or isinstance(testcase_value, dict):
                    # if it is a list or dict, skip directly
                    continue
                testcase_value = str(testcase_value).replace(" ", "").strip()
                for scenario_key, scenario_value in scenario.items():
                    # both are time variables
                    if not judge_same(scenario_key, testcase_key):
                        continue
                    if is_time_key(testcase_key) and is_time_key(scenario_key):
                        if ":" not in testcase_value and ":" not in scenario_value:  # cases like time is "上市首日" (first listing day), handled as enum variables
                            for s_value in scenario_value.split(","):
                                if judge_same(testcase_value, s_value):
                                    scenario_variables[scenario_key] = 1
                                    break
                                # else: if the times conflict, do not mark the corresponding variable as covered, and keep comparing
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

                    # both are quantity variables
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
                        fulfill_all = True  # here we assume all constraints are satisfied
                        for sv in scenario_value.split(","):
                            fulfill = False
                            for num in nums:
                                if is_number(num):
                                    num = int(num) if "." not in num else int(float(num))
                                else:  # enum constraint
                                    if judge_same(num, sv):
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
                            # for a positive case, must fulfill_all; for a negative case, fulfilling any one counts as success
                            if fei and fulfill:
                                fulfill_all = True
                                break
                            if not fulfill:
                                fulfill_all = False
                                break
                        if fulfill_all:
                            scenario_variables[scenario_key] = 1

                    # both are price variables
                    elif is_price_key(testcase_key) and is_price_key(scenario_key):
                        prices = [t.strip() for t in testcase_value.split("或")]
                        fei = False
                        if "非" == scenario_value[0]:
                            scenario_value = scenario_value[1:]
                            fei = True
                        fulfill_all = True  # here we assume all constraints are satisfied
                        for sv in scenario_value.split(","):
                            fulfill = False
                            for price in prices:
                                if is_number(price):
                                    price = float(price)
                                else:
                                    if judge_same(price, sv):
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

                    # both are enum variables
                    elif not is_time_key(testcase_key) and not is_time_key(testcase_key) and not is_num_key(testcase_key) and not is_num_key(scenario_key) and not is_price_key(testcase_key) and not is_price_key(scenario_key):
                        
                        # for an enum variable in the scenario, it counts as covered if a value-similar string exists in the testcase; otherwise not
                        for s_value in scenario_value.split(","):
                            if judge_same(testcase_value, s_value):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    parser.add_argument("--method", type=str, default="ours")
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    if method == "ours_gpt":
        result = {}
        d = dataset
        for file in sorted(os.listdir("gpt_ours")):
            if d not in file:
                continue
            gpt_ours_testcases = json.load(open(f"gpt_ours/{file}", 'r', encoding='utf-8'))
            gpt_ours_testcases = [item['testcase'] for item in gpt_ours_testcases if item['testcase'] != []]
            gpt_ours_testcases = [tc for sublist in gpt_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(gpt_ours_testcases, label_testcases, metric['ours_precision'], metric['ours_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_gpt在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_gpt_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "ours_grok":
        d = dataset
        result = {}
        for file in sorted(os.listdir("grok_ours")):
            if d not in file:
                continue
            grok_ours_testcases = json.load(open(f"grok_ours/{file}", 'r', encoding='utf-8'))
            grok_ours_testcases = [item['testcase'] for item in grok_ours_testcases if item['testcase'] != []]
            grok_ours_testcases = [tc for sublist in grok_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(grok_ours_testcases, label_testcases, metric['ours_precision'], metric['ours_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_grok在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_grok_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "ours_deepseek":
        d = dataset
        result = {}
        for file in sorted(os.listdir("deepseek_ours")):
            if d not in file:
                continue
            deepseek_ours_testcases = json.load(open(f"deepseek_ours/{file}", 'r', encoding='utf-8'))
            deepseek_ours_testcases = [item['testcase'] for item in deepseek_ours_testcases if item['testcase'] != []]
            deepseek_ours_testcases = [tc for sublist in deepseek_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(deepseek_ours_testcases, label_testcases, metric['ours_precision'], metric['ours_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_deepseek在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_deepseek_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "deepseek":
        d = dataset
        result = {}
        for file in sorted(os.listdir("deepseek")):
            if d not in file:
                continue
            deepseek_testcase_file, label_testcase_file = f"deepseek/{file}", f"testcase/{dataset}.json"
            deepseek_testcases = json.load(open(deepseek_testcase_file, 'r', encoding='utf-8'))
            deepseek_testcases = [item['testcase'] for item in deepseek_testcases if item['testcase'] != []]
            deepseek_testcases = [tc for sublist in deepseek_testcases for tc in sublist]
            label_testcases = json.load(open(label_testcase_file, 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(deepseek_testcases, label_testcases, metric['llm_precision'], metric['llm_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"deepseek在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/deepseek_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "gpt":
        d = dataset
        result = {}
        for file in sorted(os.listdir("gpt")):
            if d not in file:
                continue
            gpt_testcase_file, label_testcase_file = f"gpt/{file}", f"testcase/{dataset}.json"
            gpt_testcases = json.load(open(gpt_testcase_file, 'r', encoding='utf-8'))
            gpt_testcases = [item['testcase'] for item in gpt_testcases if item['testcase'] != []]
            gpt_testcases = [tc for sublist in gpt_testcases for tc in sublist]
            label_testcases = json.load(open(label_testcase_file, 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(gpt_testcases, label_testcases, metric['llm_precision'], metric['llm_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase

        print(f"gpt在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/gpt_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "grok":
        d = dataset
        result = {}
        for file in sorted(os.listdir("grok")):
            if d not in file:
                continue
            grok_testcase_file, label_testcase_file = f"grok/{file}", f"testcase/{dataset}.json"
            grok_testcases = json.load(open(grok_testcase_file, 'r', encoding='utf-8'))
            grok_testcases = [item['testcase'] for item in grok_testcases if item['testcase'] != []]
            grok_testcases = [tc for sublist in grok_testcases for tc in sublist]
            label_testcases = json.load(open(label_testcase_file, 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(grok_testcases, label_testcases, metric['llm_precision'], metric['llm_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase

        print(f"grok在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/grok_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "llm4fin":
        d = dataset
        result = {}
        for file in sorted(os.listdir("llm4fin")):
            if d not in file:
                continue
            llm4fin_testcase_file, label_testcase_file = f"llm4fin/{file}", f"testcase/{dataset}.json"
            llm4fin_testcases = json.load(open(llm4fin_testcase_file, 'r', encoding='utf-8'))
            llm4fin_testcases = [tc for sublist in llm4fin_testcases for tc in sublist]
            label_testcases = json.load(open(label_testcase_file, 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(llm4fin_testcases, label_testcases, metric['cmp_precision'], metric['cmp_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase

        print(f"llm4fin在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/llm4fin_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    elif method == "expert":
        d = dataset
        result = {}
        for file in sorted(os.listdir("expert")):
            if d not in file:
                continue
            expert_testcase_file, label_testcase_file = f"expert/{file}", f"testcase/{dataset}.json"
            expert_testcases = json.load(open(expert_testcase_file, 'r', encoding='utf-8'))
            label_testcases = json.load(open(label_testcase_file, 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(expert_testcases, label_testcases, metric['exp_precision'], metric['exp_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase

        print(f"expert在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/expert_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "ours_gpt_only":
        result = {}
        d = dataset
        for file in sorted(os.listdir("gpt_only_ours")):
            if d not in file:
                continue
            gpt_ours_testcases = json.load(open(f"gpt_only_ours/{file}", 'r', encoding='utf-8'))
            gpt_ours_testcases = [item['testcase'] for item in gpt_ours_testcases if item['testcase'] != []]
            gpt_ours_testcases = [tc for sublist in gpt_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(gpt_ours_testcases, label_testcases, metric['only_precision'], metric['only_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_gpt_only在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_gpt_only_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "ours_grok_only":
        d = dataset
        result = {}
        for file in sorted(os.listdir("grok_only_ours")):
            if d not in file:
                continue
            grok_ours_testcases = json.load(open(f"grok_only_ours/{file}", 'r', encoding='utf-8'))
            grok_ours_testcases = [item['testcase'] for item in grok_ours_testcases if item['testcase'] != []]
            grok_ours_testcases = [tc for sublist in grok_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(grok_ours_testcases, label_testcases, metric['only_precision'], metric['only_recall '])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_grok_only在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_grok_only_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    elif method == "ours_deepseek_only":
        d = dataset
        result = {}
        for file in sorted(os.listdir("deepseek_only_ours")):
            if d not in file:
                continue
            deepseek_ours_testcases = json.load(open(f"deepseek_only_ours/{file}", 'r', encoding='utf-8'))
            deepseek_ours_testcases = [item['testcase'] for item in deepseek_ours_testcases if item['testcase'] != []]
            deepseek_ours_testcases = [tc for sublist in deepseek_ours_testcases for tc in sublist]
            label_testcases = json.load(open(f"testcase/{dataset}.json", 'r', encoding='utf-8'))
            precision_testcase, recall_testcase, f1_testcase = eval_testcase(deepseek_ours_testcases, label_testcases, metric['ours_precision'], metric['ours_recall'])
            result[dataset] = {}
            result[dataset]["precision_testcase"] = precision_testcase
            result[dataset]["recall_testcase"] = recall_testcase
            result[dataset]['f1_testcase'] = f1_testcase
        print(f"ours_deepseek_only在{d}上的结果，testcase precision: {result[d]['precision_testcase']}, testcase recall: {result[d]['recall_testcase']}, testcase f1: {result[d]['f1_testcase']}")
        json.dump(result, open(f"log/ours_deepseek_only_prf_{d}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    