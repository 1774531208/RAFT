import json
from trl.post_process import judge_seq
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk import edit_distance
import sys
import copy
sys.setrecursionlimit(10000)


log = open(f"eval_trl.log", "w", encoding="utf-8")
def transfer_labels(labels):
    """
    Reformat labels into a form that is convenient for computation.
    """

    for i, label in enumerate(labels):
        rules = label.split("rule")
        for idx, label in enumerate(rules[1:]):
            label = label.split("or relation")[0].strip()
            before, after = label.split("\nthen")
            if "状态" in after:
                after = after.replace("状态", "结果状态")
            if "结果结果状态" in after:
                after = after.replace("结果结果状态", "结果状态")
            label = before + "\nthen" + after
            label = label.replace("\nthen", " and")
            # now only "if" remains, and 状态 has been changed to 结果状态
            
            resultstatus = []
            result = "成功"
            ls = label.split(" ")
            j = 0
            while j < len(ls):
                l = ls[j]
                if l == "结果状态":
                    resultstatus.append(ls[j:j+3])
                    a, b = ls[:j], ls[j+3:]
                    a = a[:-1]
                    ls = a + b
                elif l == "结果":
                    result = ls[j+2]
                    a, b = ls[:j], ls[j+3:]
                    a = a[:-1]
                    ls = a + b
                else:
                    j += 1
            label = " ".join(ls)
            label += f"\nthen 结果 = {result} and "
            for l in resultstatus:
                label += " ".join(l) + " and "
            label = label[:-5]
            label = label.replace("is", "=")
            label = judge_seq(label)
            rules[idx+1] = label
        labels[i] = "\nrule ".join(rules).strip()
    return labels




threshold = 0.9
def str_same_edit_distance(str1, str2):
    # edit distance
    distance = edit_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return True
    similarity = 1 - distance / max_len
    return similarity >= threshold



def lcs(s1, s2):
    # longest common subsequence
    l1, l2 = len(s1), len(s2)
    dp = [[0 for _ in range(l2+1)] for _ in range(l1+1)]
    for i in range(l1):
        for j in range(l2):
            if len(s1[i]) == 0 and len(s2[j]) == 0 and s1[i] == s2[j] or str_same_edit_distance(s1[i], s2[j]):
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[l1][l2]



def compute_accuracy(preds, labels):
    p, r, f = 0, 0, 0
    for pred, label in zip(preds, labels):
        tp = lcs(pred, label)
        precision = tp / len(pred) if len(pred) > 0 else 0
        recall = tp / len(label) if len(label) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        p += precision
        r += recall
        f += f1
        log.write(f"{precision} {recall} {f1}\n{label}\n\n{pred}\n\n")
    return p / len(preds), r / len(preds), f / len(preds)

def compute_token_accuracy(preds, labels):
    return compute_accuracy(preds, labels)

def compute_word_accuracy(preds, labels):
    for i, pred in enumerate(preds):
        ls = []
        for line in pred.split("\n"):
            words = line.split(" ")
            ls.extend(words)
        preds[i] = ls
    for i, label in enumerate(labels):
        ls = []
        for line in label.split("\n"):
            words = line.split(" ")
            ls.extend(words)
        labels[i] = ls
    return compute_accuracy(preds, labels)


def compute_structure_accuracy(preds):
    acc = 0
    for pred in preds:
        try:
            stage = 0
            compliance = True
            for line in pred.split("\n"):
                if stage == 0:
                    stage = 1
                    if "rule" in line:
                        ...
                    else:
                        compliance = False
                        break
                elif stage == 1:
                    stage = 2
                    if "if" in line:
                        words = line.split(" ")[1:]
                        i = 0
                        while i < len(words):
                            if words[i+1] in ["=", "!=", "<", "<=", ">", ">=", "in", "notin"] and (i+3 < len(words) and words[i+3] == "and" or i+3 >= len(words)) or words[i+1] == "%" and (i+5 < len(words) and words[i+5] == "and" or i+5 >= len(words)):
                                if words[i+1] == "%":
                                    i += 6
                                else:
                                    i += 4
                            else:
                                compliance = False
                                break
                    else:
                        compliance = False
                        break
                else:
                    stage = 0
                    if "then" in line:
                        words = line.split(" ")[1:]
                        i = 0
                        while i < len(words):
                            if words[i+1] in ["=", "!="] and (i+3 < len(words) and words[i+3] == "and" or i+3 >= len(words)):
                                i += 4
                            else:
                                compliance = False
                                break
                    else:
                        compliance = False
                        break
            if compliance:
                acc += 1
        except Exception:
            ...
    return acc / len(preds)



def compute_cumulative_bleu(preds, labels):
    """
    BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of machine translation, inspired by precision metrics from information retrieval. Its principle is relatively simple: it scores by comparing the overlap between the machine translation output and a human reference translation. The basic steps and principles of BLEU are as follows:

    1. Matching: BLEU first checks which words or phrases (n-grams, where n can be 1, 2, 3, 4, etc., corresponding to unigrams, bigrams, trigrams, etc.) appear in the machine translation output, and counts how many times they appear in the reference translation.
    2. Precision: For each n-gram, BLEU computes a precision score, i.e. the ratio of the number of times the n-gram appears in the machine translation output to the maximum number of times it appears in the reference translation (accuracy). This step is performed for each n-gram separately.
    3. Weighted Geometric Mean: BLEU weights the precision scores of n-grams of different lengths and computes their geometric mean. Typically, BLEU gives longer n-grams higher weight because they better reflect translation accuracy.
    5. BP (Brevity Penalty): BLEU also penalizes the length of the machine translation output. If the machine translation output is shorter than the reference, a length penalty factor BP is applied, BP = e^(1 - r/c), where r is the reference length and c is the predicted length. The purpose of BP is to penalize overly short translations, because they may omit important information.
    6. Final score: The final BLEU score is the product of the modified precision and BP. This score ranges from 0 to 1, where 1 indicates a perfect match and 0 indicates no match at all.

    BLEU's main advantages are that it is simple, fast, and can be applied at scale. However, it has also received criticism because it does not consider grammatical or semantic correctness of the language, nor does it handle language flexibility and diversity well. In addition, BLEU assigns equal weight to different n-grams, which may not be appropriate in all cases, since different n-grams may have different importance in translation quality assessment. Nevertheless, BLEU remains one of the most commonly used automatic evaluation metrics in machine translation.
    """
    cumulative_bleu = []
    for i, pred in enumerate(preds):
        ls = []
        for line in pred.split("\n"):
            words = line.split(" ")
            ls.extend(words)
        preds[i] = ls
    for i, label in enumerate(labels):
        ls = []
        for line in label.split("\n"):
            words = line.split(" ")
            ls.extend(words)
        labels[i] = ls
    for pred, label in zip(preds, labels):
        if len(pred) == 0:
            cumulative_bleu.append(0)
            continue
        bleu = sentence_bleu([label], pred, weights=(0.25, 0.25, 0.25, 0.25))
        cumulative_bleu.append(bleu)
    return sum(cumulative_bleu) / len(cumulative_bleu)


def compute_rouge_f1(predictions, labels):
    # metrics include rouge-1, rouge-2, rouge-l.
    # rouge-1 etc. compute not only accuracy but also recall and f1 score.
    # rouge-l optimizes n-gram precision into a longest-common-subsequence computation.
    r = Rouge(metrics=["rouge-l"])
    rouge_scores = []
    for pred, label in zip(predictions, labels):
        if pred == "":
            rouge_scores.append(0)
            continue
        rouge_score = r.get_scores(pred, label)
        # F1 score
        rouge_scores.append(rouge_score[0]['rouge-l']['f'])
    return sum(rouge_scores) / len(rouge_scores)


def compute_semantic_similarity(preds, labels):
    """
    The basic idea: given some data of conditions and results, run inference with the two rules separately and check whether the results are the same. For example, given n data items in total, compute accuracy = m/n, where m is the number of data items whose inference result is correct.
    """
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        if len(pred) == 0 or len(label) == 0:
            continue
        total += 1
        if pred == label:
            correct += 1
    return correct / total if total > 0 else 0



def compute_semantic_similarity(preds, labels, testcases):
    i = 0
    correct_rule = 0
    correct_cases = 0
    while i < len(preds):
        local_correct_cases = 0
        pred, label, tcs = preds[i], labels[i], testcases[i]
        pred_constraints, label_constraints = [], []

        pred_constraint = []
        for line in pred.split("\n"):
            line = line.strip()
            if "if" in line or "then" in line:
                words = line.split(" ")[1:]
                j = 0
                while j < len(words):
                    if j+2 >= len(words):
                        break
                    key, op, values = words[j], words[j+1], [words[j+2]]
                    j += 3
                    while j < len(words) and words[j] != "and":
                        values.append(words[j])
                        j += 1
                    pred_constraint.append((key, op, values))
                if "then" in line:
                    pred_constraints.append(pred_constraint)
                    pred_constraint = []
        
        label_constraint = []
        for line in label.split("\n"):
            line = line.strip()
            if "if" in line or "then" in line:
                words = line.split(" ")[1:]
                j = 0
                while j < len(words):
                    key, op, values = words[j], words[j+1], [words[j+2]]
                    j += 3
                    while j < len(words) and words[j] != "and":
                        values.append(words[j])
                        j += 1
                    label_constraint.append((key, op, values))
                if "then" in line:
                    label_constraints.append(label_constraint)
                    label_constraint = []

        # pred_constraints/label_constraints: List[List[(key, op, values)]]

        for tc in tcs:
            tc_keys = list(tc.keys())
            result = "成功"
            if "结果" in tc_keys:
                result = tc["结果"]
                tc_keys.remove("结果")

            # first check pred
            pred_sat = False
            for pc in pred_constraints:  # rule 1, rule 2, ...
                rule_sat = True
                tc_key_cover = [False for _ in range(len(tc_keys))]
                for key, op, values in pc:  # key-op-values of rule 1
                    if key not in tc_keys:
                        continue
                    idx = tc_keys.index(key)
                    tc_key_cover[idx] = True
                    real_value = tc[key]
                    if isinstance(real_value, str) and isnumber(real_value):
                        if "." in real_value:
                            real_value = float(real_value)
                        else:
                            real_value = int(real_value)
                    
                    if len(values) == 3:
                        # take the remainder
                        if op == "%":
                            v1, v2 = values[0], values[2]
                            if not isnumber(v1) or not isnumber(v2):
                                rule_sat = False
                                break
                            v1, v2 = int(v1), int(v2)
                            if not isinstance(real_value, int):
                                rule_sat = False
                                break
                            if real_value % v1 != v2:
                                rule_sat = False
                                break

                    else:
                        value = values[0]
                        if isinstance(value, str) and isnumber(value):
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        if op == "=" or op == "==":
                            if isinstance(real_value, str) and isinstance(value, str):
                                if not str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                            else:
                                if real_value != value:
                                    rule_sat = False
                                    break
                        elif op == "!=":
                            if isinstance(real_value, str) and isinstance(value, str):
                                if str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                            else:
                                if real_value == value:
                                    rule_sat = False
                                    break
                        elif op == "<":
                            if isnumber(real_value) and isnumber(value) and real_value >= value:
                                rule_sat = False
                                break
                        elif op == "<=":
                            if isnumber(real_value) and isnumber(value) and real_value > value:
                                rule_sat = False
                                break
                        elif op == ">":
                            if isnumber(real_value) and isnumber(value) and real_value <= value:
                                rule_sat = False
                                break
                        elif op == ">=":
                            if isnumber(real_value) and isnumber(value) and real_value < value:
                                rule_sat = False
                                break
                        elif op == "in":
                            if "[" in value and "]" in value:
                                if "-" not in real_value and not time_in(real_value, value):
                                    rule_sat = False
                                    break
                                elif "-" in real_value and real_value not in value:
                                    rule_sat = False
                                    break
                            else:
                                if not str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                        elif op == "notin":
                            if "[" in value and "]" in value:
                                if "-" not in real_value and time_in(real_value, value):
                                    rule_sat = False
                                    break
                                elif "-" in real_value and real_value in value:
                                    rule_sat = False
                                    break
                            else:
                                if str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                if rule_sat:
                    pred_sat = True
                    break

            # then check label
            label_sat = False
            for lc in label_constraints:  # rule 1, rule 2, ...
                rule_sat = True
                tc_key_cover = [False for _ in range(len(tc_keys))]
                for key, op, values in lc:  # key-op-values of rule 1
                    if key not in tc_keys:
                        continue
                    idx = tc_keys.index(key)
                    tc_key_cover[idx] = True
                    real_value = tc[key]
                    if isinstance(real_value, str) and isnumber(real_value):
                        if "." in real_value:
                            real_value = float(real_value)
                        else:
                            real_value = int(real_value)
                    
                    if len(values) == 3:
                        # take the remainder
                        if op == "%":
                            v1, v2 = int(values[0]), int(values[2])
                            if not isinstance(real_value, int):
                                rule_sat = False
                                break
                            if real_value % v1 != v2:
                                rule_sat = False
                                break

                    else:
                        value = values[0]
                        if isinstance(value, str) and isnumber(value):
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        if op == "=" or op == "==":
                            if isinstance(real_value, str) and isinstance(value, str):
                                if not str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                            else:
                                if real_value != value:
                                    rule_sat = False
                                    break
                        elif op == "!=":
                            if isinstance(real_value, str) and isinstance(value, str):
                                if str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                            else:
                                if real_value == value:
                                    rule_sat = False
                                    break
                        elif op == "<":
                            if isnumber(real_value) and isnumber(value) and real_value >= value:
                                rule_sat = False
                                break
                        elif op == "<=":
                            if isnumber(real_value) and isnumber(value) and real_value > value:
                                rule_sat = False
                                break
                        elif op == ">":
                            if isnumber(real_value) and isnumber(value) and real_value <= value:
                                rule_sat = False
                                break
                        elif op == ">=":
                            if isnumber(real_value) and isnumber(value) and real_value < value:
                                rule_sat = False
                                break
                        elif op == "in":
                            if "[" in value and "]" in value:
                                if "-" not in real_value and not time_in(real_value, value):
                                    rule_sat = False
                                    break
                                elif "-" in real_value and real_value not in value:
                                    rule_sat = False
                                    break
                            else:
                                if not str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                        elif op == "notin":
                            if "[" in value and "]" in value:
                                if "-" not in real_value and time_in(real_value, value):
                                    rule_sat = False
                                    break
                                elif "-" in real_value and real_value in value:
                                    rule_sat = False
                                    break
                            else:
                                if str_same_edit_distance(real_value, value):
                                    rule_sat = False
                                    break
                if rule_sat:
                    label_sat = True
                    break
            if pred_sat == label_sat:
                local_correct_cases += 1

        i += 1
        correct_cases += local_correct_cases
        if local_correct_cases == len(tcs):
            correct_rule += 1
        # exit(0)

    return correct_rule / len(preds), correct_cases / sum([len(tcs) for tcs in testcases])



def time_in(t, t_range):
    # t: str, 9:00, 
    # t_range: str, [9:00-10:30,11:00-12:00] or [9:00-10:30]
    def to_minutes(time_str):
        """Convert a time string (e.g. "9:00") to the number of minutes."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    # convert the target time to minutes
    t_min = to_minutes(t)
    
    # iterate over each time range and check whether it contains the target time
    for period in t_range[1:-1].split(","):
        period = period.strip()
        start_str, end_str = period.split('-')
        start_min = to_minutes(start_str)
        end_min = to_minutes(end_str)
        if start_min <= t_min <= end_min:
            return True
    # none of the ranges contain the target time
    return False
    



def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_trl(trls, name):
    preds = [t['predict'] for t in trls]
    labels = [t['answer'] for t in trls]
    labels = transfer_labels(labels)
    
    # token accuracy
    p, r, f = compute_token_accuracy(preds, labels)

    # bleu
    bleu = compute_cumulative_bleu(copy.deepcopy(preds), copy.deepcopy(labels))

    # rouge
    rouge = compute_rouge_f1(preds, labels)

    # word accuracy
    p_c, r_c, f_c = compute_word_accuracy(preds, labels)

    print(f"LLM: {name}\nToken ACC: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, BLEU: {bleu:.4f}, ROUGE: {rouge:.4f}\nword ACC: Precision: {p_c:.4f}, Recall: {r_c:.4f}, F1: {f_c:.4f}\n", flush=True)

    # semantic similarity
    preds = [t['predict'] for t in trls]
    labels = [t['answer'] for t in trls]
    labels = transfer_labels(labels)
    testcases = [t['testcase'] for t in trls]
    semantic_rule_acc, semantic_case_acc = compute_semantic_similarity(preds, labels, testcases)

    print(f"LLM: {name}\nSemantic Similarity: Rule ACC: {semantic_rule_acc:.4f}, Case ACC: {semantic_case_acc:.4f}\n", flush=True)





if __name__ == "__main__":
    llm = ["deepseek", "grok", "gpt"]
    doc = ["sse_trading_rules", "szse_bond_trading_rules", "szse_fund_trading_and_redemption"]
    for l in llm:
        data = []
        log.write(f"Processing LLM: {l}\n")
        for d in doc:
            res = json.load(open(f"result/postprocess_{l}_{d}.json", "r", encoding="utf-8"))
            eval_data = json.load(open(f"eval_data/{d}_eval.json", "r", encoding="utf-8"))
            for i in range(len(res)):
                res[i]['testcase'] = eval_data[i]['testcase']
            data.extend(res)
        eval_trl(data, name=l)