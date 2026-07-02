import json
import numpy as np

# Column order and grouping follow the paper table (fig/table.png).
# Four groups are computed from the raw logs in log/ (same logic as before):
#   Domain Experts -> expert
#   E2E LLMs       -> mean of deepseek / gpt / grok
#   LLM4Fin        -> llm4fin
#   RAFT           -> mean of ours_deepseek / ours_gpt / ours_grok
# Three baseline groups (ACC-Fra, MSG-Spec, RAFT+HITL) have no raw logs, so
# their per-dataset metrics are hardcoded from the values reported in
# fig/table.png. BSC = Business Scenario Coverage (the "rc" metric in the logs).

LLMS = ["deepseek", "gpt", "grok"]
DATASETS = list(range(1, 7))
METRICS = ["Precision", "Recall", "F1", "BSC"]


def read_prf(method, i):
    d = json.load(open(f"log/{method}_prf_dataset{i}.json", "r", encoding="utf-8"))[f"dataset{i}"]
    return d["precision_testcase"], d["recall_testcase"], d["f1_testcase"]


def read_rc(method, i):
    return json.load(open(f"log/{method}_rc_dataset{i}.json", "r", encoding="utf-8"))[f"dataset{i}"]


def single_method(method):
    m = {k: [] for k in METRICS}
    for i in DATASETS:
        p, r, f = read_prf(method, i)
        m["Precision"].append(p * 100)
        m["Recall"].append(r * 100)
        m["F1"].append(f * 100)
        m["BSC"].append(read_rc(method, i) * 100)
    return m


def averaged_methods(methods):
    m = {k: [] for k in METRICS}
    for i in DATASETS:
        ps, rs, fs, cs = [], [], [], []
        for method in methods:
            p, r, f = read_prf(method, i)
            ps.append(p)
            rs.append(r)
            fs.append(f)
            cs.append(read_rc(method, i))
        m["Precision"].append(np.mean(ps) * 100)
        m["Recall"].append(np.mean(rs) * 100)
        m["F1"].append(np.mean(fs) * 100)
        m["BSC"].append(np.mean(cs) * 100)
    return m


# Hardcoded from fig/table.png (percentages, dataset1..dataset6).
ACC_FRA = {
    "Precision": [41.8, 48.9, 46.4, 52.7, 58.5, 43.9],
    "Recall":    [36.5, 39.9, 34.3, 45.7, 49.8, 31.5],
    "F1":        [38.9, 43.9, 39.0, 49.0, 53.8, 36.7],
    "BSC":       [37.9, 29.8, 36.4, 36.4, 61.6, 31.3],
}
MSG_SPEC = {
    "Precision": [64.2, 58.7, 70.5, 67.8, 75.1, 60.3],
    "Recall":    [55.8, 50.4, 62.1, 71.3, 68.9, 48.5],
    "F1":        [59.7, 54.2, 66.0, 69.5, 71.9, 53.7],
    "BSC":       [61.3, 58.9, 68.4, 64.7, 72.3, 51.8],
}
RAFT_HITL = {
    "Precision": [95.5, 95.4, 96.5, 92.0, 98.5, 95.8],
    "Recall":    [95.7, 94.2, 95.7, 98.0, 97.1, 98.5],
    "F1":        [95.6, 94.8, 96.1, 94.9, 97.8, 97.1],
    "BSC":       [90.5, 96.2, 90.9, 84.0, 89.8, 89.8],
}


def build_table():
    return {
        "Domain Experts": single_method("expert"),
        "E2E LLMs": averaged_methods(LLMS),
        "ACC-Fra": ACC_FRA,
        "MSG-Spec": MSG_SPEC,
        "LLM4Fin": single_method("llm4fin"),
        "RAFT": averaged_methods([f"ours_{llm}" for llm in LLMS]),
        "RAFT+HITL": RAFT_HITL,
    }


SHORT = {"Precision": "Pre.", "Recall": "Rec.", "F1": "F1", "BSC": "BSC"}


def main():
    table = build_table()
    methods = list(table.keys())

    # Two-level header: methods span their 4 metric columns; second row repeats Pre./Rec./F1/BSC.
    top = ["Datasets"] + [name if i == 0 else "" for name in methods for i in range(len(METRICS))]
    sub = [""] + [SHORT[metric] for _ in methods for metric in METRICS]
    rows = [top, sub]

    # One row per dataset, then Average and Variance.
    row_labels = [str(i) for i in DATASETS] + ["Average", "Variance"]
    for r, label in enumerate(row_labels):
        row = [label]
        for name in methods:
            for metric in METRICS:
                vals = table[name][metric]
                if label == "Average":
                    v = float(np.mean(vals))
                elif label == "Variance":
                    v = float(np.var(vals))
                else:
                    v = vals[r]
                row.append(str(round(v, 2)))
        rows.append(row)

    with open("fig/table.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(",".join(cell for cell in row) for row in rows) + "\n")
    print("Table saved to fig/table.csv")


if __name__ == "__main__":
    main()
