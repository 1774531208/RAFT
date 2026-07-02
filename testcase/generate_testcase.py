import json
import z3
import re
from collections import OrderedDict
import copy


"""
A->B truth table
A  B  A->B
----------
0  0   1
0  1   1
1  0   0   condition true, conclusion false -> false
1  1   1   condition true, conclusion true -> true

A^B -> C^D truth table
A  B  C  D  A^B  C^D  A^B -> C^D
-----------------------------------
0  0  0  0   0    0        1
0  0  0  1   0    0        1
0  0  1  0   0    0        1
0  0  1  1   0    1        0
0  1  0  0   0    0        1
0  1  0  1   0    0        1
0  1  1  0   0    0        1
0  1  1  1   0    1        0
1  0  0  0   0    0        1
1  0  0  1   0    0        1
1  0  1  0   0    0        1
1  0  1  1   0    1        0
1  1  0  0   1    0        0
1  1  0  1   1    0        0
1  1  1  0   1    0        0   condition true, false if any conclusion is false
1  1  1  1   1    1        1   condition true, conclusion true -> true
"""

"""
When generating test cases, we do not consider the case where the condition is false, because that is meaningless.
In the result part, we only generate counterexamples for numeric-type constraints such as "时间" (time), "数量" (quantity), "价格" (price) and for "操作" (action), while also taking care to change the status information; other enumerated variables stay unchanged.
"""






def find_word(s, word):
    """Find all positions where word occurs in the string s."""
    locs = [s.find(word)]
    while locs[-1] != -1:
        locs.append(s.find(word, locs[-1]+1))
    return locs[:-1]


def time_preprocess(time):
    """
    Process a time string: determine whether it contains a time period like 9:00-10:00, or expressions like "before"/"after". If so, return True and the processed time period; otherwise return False and an empty string.
    Args:
        time: time string, e.g. "9:00-10:00 of each trading day"
    Returns:
        valid: whether it is a valid time period
        numerical_time: the processed time period, e.g. "9:00-10:00,11:00-12:00" or ""
    """
    time_vals = re.findall(r"\d+:\d+", time)  # all time values like 9:00
    vals_locs = [time.find(time_val) for time_val in time_vals]  # positions of the time values within time
    loc_before = sorted(find_word(time, "<") + find_word(time, "<="))  # positions of comparison words like earlier-than / later-than / until within time
    loc_after = sorted(find_word(time, ">") + find_word(time, ">="))
    loc_between = find_word(time, "-")
    locs = sorted(loc_before + loc_after + loc_between)
    if time_vals:
        time_vals = ["0" + time_val if len(time_val) == 4 else time_val for time_val in time_vals]
        t = ""
        # consider three time cases: "9:00 to 10:00", "after 9:00 / later than 9:00", "before 9:00 / earlier than 9:00"; other cases are copied verbatim
        if len(vals_locs) != len(loc_before) + len(loc_after) + 2*len(loc_between):
            return False, ""
        p = 0
        valid = True
        for loc in locs:
            if loc in loc_before:
                if time[loc:loc+1] == "<":
                    t += f"00:00:00-{time_vals[p]}:00,"
                    p += 1
                else:
                    valid = False
                    break
            elif loc in loc_after:
                if time[loc:loc+1] == ">":
                    t += f"{time_vals[p]}:00-24:00:00,"
                    p += 1
                else:
                    valid = False
                    break
            else:
                if p+1 < len(vals_locs) and vals_locs[p] < loc and vals_locs[p+1] > loc:
                    t += f"{time_vals[p]}:00-{time_vals[p+1]}:00,"
                    p += 2
                else:
                    valid = False
                    break
        if valid:
            return True, t[:-1]
        else:
            return False, ""
    else:
        return False, ""


def generate_time_testcase(consequence: list):
    """
    Process time-type constraints and generate the corresponding test cases.
    Args:
        consequence: a time constraint
    Returns:
        time_testcase: list of time test cases; generated test cases must be appended to this list
    """
    time_testcase = []
    time = consequence[2] if consequence[1] in ["=", "==", "!=", "in", "notin"] else consequence[1] + consequence[2]
    time.replace("不晚于", "早于").replace("不早于", "晚于")
    valid, numerical_time = time_preprocess(time)
    if valid:  # time is a period like 9:00-10:00, generate the corresponding test cases
        # array like [[9:00-10:00], ] converted to [[09:00-10:00], ], then generate counterexamples
        time_testcase.append([consequence[0], numerical_time])
        # generate counterexamples
        time_list = []
        for t in numerical_time.split(","):
            time_list.extend(t.split("-"))
        for i in range(len(time_list)):
            if len(time_list[i]) == 4:
                time_list[i] = "0" + time_list[i]
        new_time_list = []
        begin = "00:00:00"
        i = 0
        while i < len(time_list):
            if time_list[i] == begin:
                begin = time_list[i+1]
                i += 2
                continue
            new_time_list.append(f"{begin}-{time_list[i]}")
            begin = time_list[i+1]
            i += 2
        if begin != "24:00:00":
            new_time_list.append(f"{begin}-24:00:00")
        time_testcase.append([consequence[0], ",".join(new_time_list)])
        if consequence[1] in ["!=", "notin"]:
            time_testcase = time_testcase[1:] + time_testcase[:1]
    else:  # no numeric time, may be things like "当日" (that day) or "竞买日前" (before the auction day)
        if "<" in time or ">" in time or "<=" in time or ">=" in time:  # generate positive and negative test cases
            if "<" in time or "<=" in time:
                if "<=" in time:
                    time = time[2:]
                else:
                    time = time[1:]
                time_testcase.append([consequence[0], time])
                time_testcase.append([consequence[0], time + "后"])
            elif ">" in time or ">=" in time:
                if ">=" in time:
                    time = time[2:]
                else:
                    time = time[1:]
                time_testcase.append([consequence[0], time])
                time_testcase.append([consequence[0], time + "前"])
        else:  # no special handling needed
            time_testcase.append([consequence[0], time])
            time_testcase.append([consequence[0], "非" + time])
            if consequence[1] in ["!=", "notin"]:
                time_testcase = time_testcase[1:] + time_testcase[:1]
    return time_testcase

def judge_op(value):
    if "不低于" in value or "达到" in value or "以上" in value:
        return ">="
    if "不高于" in value or "以下" in value or "不超过" in value or "以内" in value:
        return "<="
    if "低于" in value or "未达到" in value or "不足" in value or "小于" in value:
        return "<"
    if "高于" in value or "超过" in value or "优于" in value or "大于" in value:
        return ">"
    if "不等于" in value:
        return "!="
    return "="


def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def generate_consequence_z3_expr(consequences_without_time, z3_variables):
    """
    Convert all consequences into z3 expressions.
    """
    # avoid duplicate keys
    keys = []
    for consequence in consequences_without_time:
        if consequence[0] not in keys:
            keys.append(consequence[0])
        elif not is_num_key(consequence[0]) and not is_price_key(consequence[0]):
            i = 2
            while consequence[0] + str(i) in keys:
                i += 1
            consequence[0] = consequence[0] + str(i)
            keys.append(consequence[0])

    z3_expr = []
    for consequence in consequences_without_time:
        # x is a z3 variable
        if consequence[0] in z3_variables:
            x = z3_variables[consequence[0]]
        else:
            # special handling: +-100% of the previous closing price, no more than 100 basis points above the matched most-recent trade price, arithmetic expressions (e.g. previous closing price - 100 yuan * this repayment ratio)
            if "%" == consequence[1]:
                x = z3.Int(consequence[0])
            elif "基点" in consequence[2] or any([op in consequence[2:] for op in ["+", "-", "*", "/", "="]]) or (not is_num_key(consequence[0]) and not is_price_key(consequence[0])):
                x = z3.String(consequence[0])
            elif re.findall(r"\d+\.\d+", consequence[2]):
                x = z3.Real(consequence[0])
            elif re.findall(r"\d+", consequence[2]):
                x = z3.Int(consequence[0])
            else:
                x = z3.String(consequence[0])
            z3_variables[consequence[0]] = x
        
        # quantity and price need special handling
        if is_num_key(consequence[0]):
            if len(consequence) == 3 and isnumber(consequence[2]):
                num = float(consequence[2]) if "." in consequence[2] else int(consequence[2])
                op = consequence[1]
                exp = None
                if op == ">=":
                    exp = x >= num
                elif op == "<=":
                    exp = x <= num
                elif op == "<":
                    exp = x < num
                elif op == ">":
                    exp = x > num
                elif op == "=":
                    exp = x == num
                elif op == "!=":
                    exp = x != num
                else:
                    exp = x == num
                z3_expr.append(exp)
            elif len(consequence) == 5 and isnumber(consequence[2]) and isnumber(consequence[4]) and consequence[1] == "%" and consequence[3] in ['=', '=='] and "." not in consequence[2] and "." not in consequence[4]:
                num1 = int(consequence[2])
                num2 = int(consequence[4])
                exp = x % num1 == num2
                z3_expr.append(exp)
            else:
                del z3_variables[consequence[0]]


        elif is_price_key(consequence[0]):
            if len(consequence) == 3 and isnumber(consequence[2]):
                num = float(consequence[2]) if "." in consequence[2] else int(consequence[2])
                op = consequence[1]
                exp = None
                if op == ">=":
                    exp = x >= num
                elif op == "<=":
                    exp = x <= num
                elif op == "<":
                    exp = x < num
                elif op == ">":
                    exp = x > num
                elif op == "=":
                    exp = x == num
                elif op == "!=":
                    exp = x != num
                else:
                    exp = x == num
                z3_expr.append(exp)
            elif len(consequence) == 5 and isnumber(consequence[2]) and isnumber(consequence[4]) and consequence[1] == "%" and consequence[3] in ['=', '=='] and "." not in consequence[2] and "." not in consequence[4]:
                num1 = int(consequence[2])
                num2 = int(consequence[4])
                exp = x % num1 == num2
                z3_expr.append(exp)
            else:
                del z3_variables[consequence[0]]
        elif "操作" in consequence[0] and (consequence[0][2:] == "" or consequence[0][2:].isdigit() and int(consequence[0][2:]) < 5):
            if consequence[1] in ["=", "==", "in"]:
                z3_expr.append(x == consequence[2])
            elif consequence[1] in ["!=", "notin"]:
                z3_expr.append(x != consequence[2])
        else:
            del z3_variables[consequence[0]]
    return z3_expr


def safe_decode(value):
    s = str(value)
    # handle possible \\\\u{...}
    s = s.replace("\\\\", "\\")
    # key point: handle \u{xxxxxx}
    s = re.sub(r'\\u\{([0-9a-fA-F]+)\}', lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace("\"", "").replace("{", "").replace("}", "")
    return s

def generate_consequence_case_list(solver, z3_variables, consequence_z3_expr, consequence_case_list, index):
    """
    Recursive function that negates each item in consequence_z3_expr and takes the Cartesian product, generating all possible cases and appending them to consequence_case_list.
    Args:
        solver: z3 solver
        z3_variables: dict of z3 variables
        consequence_z3_expr: list of z3 expressions
        consequence_case_list: list of consequence test cases
        index: index of the currently processed consequence
    """
    if index == len(consequence_z3_expr):
        if solver.check() == z3.sat:
            rs = solver.model()
            consequence_case = []
            for variable in z3_variables:
                key = z3_variables[variable]
                value = rs[key]

                if isinstance(value, z3.IntNumRef):
                    value = int(value.as_long())
                elif isinstance(value, z3.RatNumRef):
                    value = float(value.as_decimal(prec=6)[:6])
                else:  # z3.SeqRef
                    # value = str(value).replace("\\\\", "\\").replace("{", "").replace("}", "").replace("\"", "").encode("utf-8").decode("unicode_escape")
                    value = safe_decode(value)
                consequence_case.append([str(key), value])
            consequence_case_list.append(consequence_case)
        return
    
    expr = consequence_z3_expr[index]
    
    solver.push()
    solver.add(expr)
    generate_consequence_case_list(solver, z3_variables, consequence_z3_expr, consequence_case_list, index+1)
    solver.pop()

    solver.push()
    solver.add(z3.Not(expr))
    generate_consequence_case_list(solver, z3_variables, consequence_z3_expr, consequence_case_list, index+1)
    solver.pop()


def post_process_blank(consequence_case_list, consequences_without_time):
    """
    Post-processing: in z3, string != "abc" produces an empty string; handle that empty value.
    """
    for case in consequence_case_list:
        for c in case:
            if c[1] == "":
                for consequence in consequences_without_time:
                    if consequence[0] == c[0]:
                        if consequence[2].startswith("不") or consequence[2].startswith("非"):
                            c[1] = consequence[2][1:]
                        else:
                            c[1] = "非" + consequence[2]
                        break

def is_time_key(key):
    if key[-1] == "日" or key[-2:] == "时间" or "期" in key:
        return True
    return False

def is_num_key(key):
    if "量" in key or "数" in key:
        return True
    return False

def is_price_key(key):
    if ("价" in key or "基准" == key or "金额" in key) and "要素" not in key and "指令" not in key and "类型" not in key and "方式" not in key:
        return True
    if any([t in key for t in ['速度', '距离', '金额', '电压', '频率', '预测精度', '功率响应']]):
        return True
    return False



def cartesian_product(nums):
    """
    Compute the Cartesian combination.
    For example: nums=[[1,2],[3,4]], treating each sublist as a group
    return: [[1,3], [1,4], [2,3], [2,4]]
    """
    if len(nums) == 0:
        return []
    if len(nums) == 1:
        return [[num] for num in nums[0]]
    sub_res = cartesian_product(nums[1:])
    res = []
    for num in nums[0]:
        for sub in sub_res:
            res.append([num] + sub)
    return res


def mydsl_to_rule(trl):
    """
    trl = {
        "rule": "停止接受买入申报的，当日不再恢复，本所另有规定的除外。",
        "answer": "rule 1\nif 时间 = 当日 and 约束 != 本所另有规定 and 事件 = 停止接受买入申报\nthen 结果状态 = 不恢复\nrule 2\nif 约束 = 本所另有规定 and 事件 = 停止接受买入申报\nthen 结果 = 不适用",
        "predict": "rule 1\nif 时间 = 当日 and 约束 != 本所另有规定 and 事件 = 停止接受买入申报\nthen 结果状态 = 不恢复\nrule 2\nif 约束 = 本所另有规定 and 事件 = 停止接受买入申报\nthen 结果 = 不适用"
    },
    rule = {
        "rule": text,
        "trl": [
            {
                "conditions": [
                    [key, op, value1, value2, ...],
                    ...
                ],
                "consequences": [
                    [key, op, value1, value2, ...],
                    ...
                ]
            }
        ]
    }
    """
    rule = {
        "rule": trl['rule']
    }
    trl = trl['predict']
    res = []
    for line in trl.split("\n"):
        l = line.strip().split(" ")
        if len(l) == 0:
            continue
        if l[0] == "rule":
            r = {}
            key_values = []
            key_values_consequences = []
        elif l[0] == "if" or l[0] == "then":
            i = 1
            while i < len(l):
                j = i
                while j < len(l) and l[j] != "and":
                    j += 1
                if j - i == 1:
                    key = "约束"
                    op = "="
                    values = [l[i]]
                else:
                    key = l[i]
                    op = l[i+1]
                    if op == "==":
                        op = "="
                    values = l[i+2:j]
                    if "true" in values:
                        values = [key]
                        key = "约束"
                        if op != "notin" and op != "!=":
                            op = "="
                        else:
                            op = "!="
                if key == "结果" or key == "结果状态":
                    key_values_consequences.append([key, op] + values)
                else:
                    key_values.append([key, op] + values)
                i = j + 1
            if l[0] == "then":
                r['conditions'] = key_values
                r['consequences'] = key_values_consequences
                # deduplicate: some trls state both the success scenario and, conversely, the failure scenario, causing duplicates
                cf = False
                for old_r in res:
                    # check
                    try:
                        old_keys, old_ops, old_values = zip(*old_r['conditions'])
                        new_keys, new_ops, new_values = zip(*r['conditions'])
                    except Exception as e:
                        print(f"Error parsing trl: the trl format is malformed, please check or regenerate!\n{trl}\n", e)
                        exit(-1)
                    if old_keys == new_keys and old_values == new_values:
                        cf = True
                        break
                if not cf:
                    res.append(r)
    rule['trl'] = res
    return rule



def generate_testcase(trls):
    """
    Generate test cases.
    """
    # rule = {
    #     "rule": text,
    #     "trl": [
    #         {
    #             "conditions": [
    #                 [key, op, value1, value2, ...],
    #                 ...
    #             ],
    #             "consequences": [
    #                 [key, op, value1, value2, ...],
    #                 ...
    #             ]
    #         }
    #     ]
    # }
    rules = []
    for trl in trls:
        rule = mydsl_to_rule(trl)
        rules.append(rule)


    for rule_i in rules:
        testcases = []
        for rule in rule_i['trl']:
            index = 1
            testcase = []
            result, resultstatus = "成功", ""
            for c in rule['consequences']:
                if c[0] == "结果":
                    result = c[2]
                elif c[0] == "结果状态":
                    resultstatus = c[2]

            # handle time on our own
            time_testcase = []
            condition_without_time = []
            for condition in rule['conditions']:
                if is_time_key(condition[0]):
                    local_time_testcase = generate_time_testcase(condition)
                    time_testcase.append(local_time_testcase)
                else:
                    if condition[1] == "in" or condition[1] == "==":
                        condition[1] = "="
                    elif condition[1] == "notin" or condition[1] == "!=":
                        condition[1] = "!="
                    condition_without_time.append(condition)
            time_testcase = cartesian_product(time_testcase)

            # quantity preprocessing: if the number of numeric quantity keys to generate is >= 2 and they differ, unify them into "数量"
            num_keys = {}
            for condition in condition_without_time:
                if is_num_key(condition[0]) and re.findall(r"\d+", condition[2]):
                    if condition[0] in num_keys:
                        num_keys[condition[0]] += 1
                    else:
                        num_keys[condition[0]] = 1
            if len(list(num_keys.keys())) >= 2:
                for condition in condition_without_time:
                    if condition[0] in num_keys:
                        condition[0] = "数量"

            # generate z3 expressions
            z3_variables = {}
            consequence_z3_expr = generate_consequence_z3_expr(condition_without_time, z3_variables)
            # recursively generate test cases using z3
            consequence_case_list = []
            solver = z3.Solver()
            for v in z3_variables.values():
                if isinstance(v, z3.ArithRef):
                    solver.add(v > 0)
            generate_consequence_case_list(solver, z3_variables, consequence_z3_expr, consequence_case_list, 0)

            for consequence in condition_without_time:
                if consequence[0] not in z3_variables:
                    for c in consequence_case_list:
                        if consequence[1] == "=":
                            c.append([consequence[0], consequence[2]])
                        elif consequence[1] == "!=":
                            c.append([consequence[0], "非" + consequence[2]])
                        else:
                            c.append([consequence[0], consequence[1] + consequence[2]])
            
            # post-processing: in z3, string != "abc" produces an empty string; handle that empty value
            post_process_blank(consequence_case_list, condition_without_time)

            # add the time test cases to the test cases
            if time_testcase:
                new_consequence_case_list = []
                for c in consequence_case_list:
                    for time_case in time_testcase:
                        new_case = c.copy()
                        for t in time_case:
                            # idx=2
                            # keys = [cc[0] for cc in new_case]
                            # if t[0] in keys:
                            #     while t[0] + str(idx) in keys:
                            #         idx += 1
                            #     t[0] = t[0] + str(idx)
                            new_case.append(t)
                        new_consequence_case_list.append(new_case)
                consequence_case_list = new_consequence_case_list


            # combine result constraints with condition constraints
            testcase_of_this_rule = []
            for c in consequence_case_list:
                new_testcase = copy.deepcopy(testcase)
                new_testcase.extend(copy.deepcopy(c))
                
                final_testcase = OrderedDict()
                
                # if a key appears in the conditions and is duplicated, name them key, key2, key3, ...
                # if a key appears in the results and is duplicated, name them 结果key, 结果key2, 结果key3, ...
                for tc in new_testcase:
                    if len(final_testcase) < len(testcase) and tc[0] in final_testcase:  # key appears in the conditions and is duplicated
                        key_index = 2
                        while tc[0] + str(key_index) in final_testcase:
                            key_index += 1
                        tc[0] = tc[0] + str(key_index)
                    final_testcase[tc[0]] = str(tc[1])
                
                if index == 1:
                    final_testcase['结果'] = result
                    if resultstatus != "":
                        final_testcase['结果状态'] = resultstatus
                else:
                    if result == "成功":
                        final_testcase['结果'] = "失败"
                        if resultstatus != "":
                            final_testcase['结果状态'] = final_testcase.get("状态", "非" + resultstatus)
                    else:
                        final_testcase['结果'] = "成功"
                        if resultstatus != "":
                            final_testcase['结果状态'] = resultstatus

                testcase_of_this_rule.append(final_testcase)
                index += 1
            testcases.extend(testcase_of_this_rule)
        rule_i['testcase'] = testcases
        # del rule_i['trl']
    return rules





if __name__ == "__main__":
    llm = ["deepseek", "grok", "gpt"]
    document = ["sse_trading_rules", "szse_bond_trading_rules", "szse_fund_trading_and_redemption"]
    for l in llm:
        for d in document:
            trls = json.load(open(f"data/postprocess_{l}_{d}.json", "r", encoding="utf-8"))
            testcase = generate_testcase(trls)
            json.dump(testcase, open(f"result/testcase_{l}_{d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)