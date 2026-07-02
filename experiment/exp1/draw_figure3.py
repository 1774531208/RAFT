import matplotlib.pyplot as plt
import numpy as np
import json


def draw_figure():
    metrics = ["Pre.", "Rec.", "F1", "BSC", "KC"]
    ours = [0 for _ in range(len(metrics))]
    llms = [0 for _ in range(len(metrics))]
    for i in range(1, 7):
        for llm in ["deepseek", "grok", "gpt"]:
            data = json.load(open(f"log/ours_{llm}_prf_dataset{i}.json", "r", encoding="utf-8"))
            ours[0] += data[f"dataset{i}"]["precision_testcase"]
            ours[1] += data[f"dataset{i}"]["recall_testcase"]
            ours[2] += data[f"dataset{i}"]["f1_testcase"]
            data = json.load(open(f"log/ours_{llm}_rc_dataset{i}.json", "r", encoding="utf-8"))
            ours[3] += data[f"dataset{i}"]
            data = json.load(open(f"log/{llm}_prf_dataset{i}.json", "r", encoding="utf-8"))
            llms[0] += data[f"dataset{i}"]["precision_testcase"]
            llms[1] += data[f"dataset{i}"]["recall_testcase"]
            llms[2] += data[f"dataset{i}"]["f1_testcase"]
            data = json.load(open(f"log/{llm}_rc_dataset{i}.json", "r", encoding="utf-8"))
            llms[3] += data[f"dataset{i}"]
    for i in range(len(ours)):
        ours[i] /= 18
        llms[i] /= 18
    data = json.load(open("log/knowledge_coverage.json", "r", encoding="utf-8"))
    ours[4] = data["ours"]
    llms[4] = (data["grok"]["coverage"] + data["deepseek"]["coverage"] + data["gpt"]["coverage"]) / 3
    for i in range(len(ours)):
        ours[i] *= 100
        llms[i] *= 100
    
    x = 4
    metrics = metrics[x:] + metrics[:x]
    ours = ours[x:] + ours[:x]
    llms = llms[x:] + llms[:x]



    # radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    new_angles = []
    for angle in angles:
        new_angle = angle + np.pi / 5*2
        if new_angle > 2 * np.pi:
            new_angle -= 2 * np.pi
        new_angles.append(new_angle)
    angles = new_angles
    ours += ours[:1]
    llms += llms[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, ours, 'o-', linewidth=2, label='K2P (M)', color="#1663A9")
    ax.fill(angles, ours, color="#1663A9", alpha=0.25)

    ax.plot(angles, llms, 'o-', linewidth=2, label='K2P (I)', color="#ff6b6b")
    ax.fill(angles, llms, color="#ff6b6b", alpha=0.25)

    ax.set_xticks(angles[:-1], metrics, fontsize=25)
    ax.set_xticklabels([])
    thetas, rs = [], []
    for i, theta in enumerate(angles[:-1]):
        if i == 0:
            thetas.append(theta - 0.45)
            rs.append(120)
        elif i == 3:
            thetas.append(theta)
            rs.append(120)
        else:
            thetas.append(theta)
            rs.append(130)
    for i, theta in enumerate(angles[:-1]):
        ax.text(
            thetas[i],   # manually adjusted angle
            rs[i],      # radius
            metrics[i],
            ha='center',
            va='center',
            fontsize=25
        )
    ax.set_yticks([20, 40, 60, 80, 100], [], fontsize=15)
    ax.set_ylim(0, 100)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.26), fontsize=20, ncol=2, handletextpad=0.2, # distance from color block to text
        columnspacing=0.2, # spacing between columns
        )


    plt.tight_layout()
    plt.savefig("fig/strategy_compare.png", dpi=300)





colors = ["#4d7cc5", "#FF6B6B", "#ffc339", "#B8B8B8"]


# def draw_figure_2(metric):
#     datasets = [f"dataset{i}" for i in range(1, 7)]
#     llms = ["grok", "deepseek", "gpt"]
#     methods = ["ours", "only"]
#     data = {ds: {m: 0 for m in methods} for ds in datasets}
#     for dataset in datasets:
#         for method in methods:
#             for llm in llms:
#                 if method == "ours":
#                     if metric == "rc":
#                         dataraw = json.load(open(f"log/ours_{llm}_rc_{dataset}.json", "r", encoding="utf-8"))
#                         data[dataset][method] += dataraw[dataset] / len(llms)
#                     else:
#                         dataraw = json.load(open(f"log/ours_{llm}_prf_{dataset}.json", "r", encoding="utf-8"))
#                         data[dataset][method] += dataraw[dataset][metric + "_testcase"] / len(llms)
#                 else:
#                     if metric == "rc":
#                         dataraw = json.load(open(f"log/ours_{llm}_only_rc_{dataset}.json", "r", encoding="utf-8"))
#                         data[dataset][method] += dataraw[dataset] / len(llms)
#                     else:
#                         dataraw = json.load(open(f"log/ours_{llm}_only_prf_{dataset}.json", "r", encoding="utf-8"))
#                         data[dataset][method] += dataraw[dataset][metric + "_testcase"] / len(llms)
#     for dataset in datasets:
#         for method in methods:
#             data[dataset][method] *= 100  # convert to percentage
    
    
#     # plotting parameters
#     x = np.arange(len(datasets))              # dataset positions
#     bar_width = 0.35                           # bar width
#     fig, ax = plt.subplots(figsize=(6, 4.5))
#     # plot each method
#     labels = ["K2P (M)", "K2P (I)"]
#     for i, method in enumerate(methods):
#         values = [data[ds][method] for ds in datasets]
#         ax.bar(
#             x + i * bar_width,
#             values,
#             width=bar_width,
#             label=labels[i],
#             color=colors[i],
#         )
#     # axes and labels
#     ax.set_xlabel("Dataset", fontsize=35)
#     if metric == "rc":
#         metric = "BSC"
#     else:
#         metric = metric.capitalize()
#     ax.set_xticks(x + bar_width / 2)
#     ax.set_xticklabels([i for i in range(1, 7)], fontsize=35)
#     ax.tick_params(axis='y', labelsize=35)
#     # plt.legend(fontsize=30)
#     ax.set_yticks(np.arange(0, 101, 20))
#     ax.set_ylim(0, 100)
#     # beautify
#     ax.grid(axis="both", linestyle="--", alpha=0.5)
#     ax.set_axisbelow(True)
#     plt.tight_layout()
#     plt.savefig(f"fig/exp2_2_{metric}.png", dpi=300)



# def draw_know():
#     data = json.load(open("log/knowledge_coverage.json", "r", encoding="utf-8"))
#     data = {
#         "K2P (M)": data["ours"] * 100,
#         "K2P (I)": (data["grok"]["coverage"] + data["deepseek"]["coverage"] + data["gpt"]["coverage"]) / 3 * 100
#     }
#     methods = list(data.keys())
#     fig, ax = plt.subplots(figsize=(2.5, 4.5))
#     x = np.arange(1)
#     width = 0.25
#     for i, method in enumerate(methods):
#         values = [data[method]]
#         ax.bar(
#             i * width,
#             values,
#             width=width,
#             label=method,
#             color=colors[i],
#         )
#     ax.set_ylim(0, 100)
#     ax.set_yticks(np.arange(0, 101, 20))
#     ax.set_xticks([width / 2])
#     ax.set_xticklabels([" "])
#     ax.set_xlabel(" ", fontsize=35)
#     ax.tick_params(axis='x', labelsize=35)
#     ax.tick_params(axis='y', labelsize=35)
#     plt.tight_layout()
#     plt.savefig("fig/exp2_2_kc.png", dpi=300)

def draw_figure_2_merged():
    colors = ["#4d7cc5", "#FF6B6B94"]
    datasets = [f"dataset{i}" for i in range(1, 7)]
    metrics = ["precision", "recall", "f1", "rc"]
    llms = ["grok", "deepseek", "gpt"]
    methods = ["ours", "only"]
    labels = ["RAFT (M)", "RAFT (I)"]

    num_datasets = len(datasets)
    gap = 0.5
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(26, 5.5))

    xticks = []
    xtick_labels = []
    base_x = 0  # current metric starting x position

    know_data = json.load(open("log/knowledge_coverage.json", "r", encoding="utf-8"))
    know_data = {
        "RAFT (M)": know_data["ours"] * 100,
        "RAFT (I)": (know_data["grok"]["coverage"] + know_data["deepseek"]["coverage"] + know_data["gpt"]["coverage"]) / 3 * 100
    }
    print("KC", know_data)

    # plot the knowledge coverage bars
    x = np.arange(1) + base_x
    for i, method in enumerate(methods):
        values = [know_data[labels[i]]]
        ax.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            color=colors[i]
        )
    center = x.mean() + bar_width / 2
    ax.text(center, 105, "KC", ha="center", va="bottom", fontsize=25)
    xticks.extend(x + bar_width / 2)
    xtick_labels.extend([" "])
    base_x += 1 + gap
    # add a vertical line to separate the different subplots
    ax.axvline(x=base_x - gap-0.08, color='black', linestyle='-', linewidth=1)



    for metric_idx, metric in enumerate(metrics):
        # -------- compute data --------
        data = {ds: {m: 0 for m in methods} for ds in datasets}

        for dataset in datasets:
            for method in methods:
                for llm in llms:
                    if method == "ours":
                        if metric == "rc":
                            dataraw = json.load(open(f"log/ours_{llm}_rc_{dataset}.json"))
                            data[dataset][method] += dataraw[dataset] / len(llms)
                        else:
                            dataraw = json.load(open(f"log/ours_{llm}_prf_{dataset}.json"))
                            data[dataset][method] += dataraw[dataset][metric + "_testcase"] / len(llms)
                    else:
                        if metric == "rc":
                            dataraw = json.load(open(f"log/ours_{llm}_only_rc_{dataset}.json"))
                            data[dataset][method] += dataraw[dataset] / len(llms)
                        else:
                            dataraw = json.load(open(f"log/ours_{llm}_only_prf_{dataset}.json"))
                            data[dataset][method] += dataraw[dataset][metric + "_testcase"] / len(llms)

        for ds in datasets:
            for m in methods:
                data[ds][m] *= 100
        
        avgdata = {m: sum(data[ds][m] for ds in datasets) / num_datasets for m in methods}
        print(metric, avgdata)

        # -------- plot bars --------
        x = np.arange(num_datasets) + base_x

        for i, method in enumerate(methods):
            values = [data[ds][method] for ds in datasets]
            ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                color=colors[i],
                label=labels[i] if metric_idx == 0 else None
            )

        # -------- x tick --------
        xticks.extend(x + bar_width / 2)
        xtick_labels.extend([str(i) for i in range(1, 7)])

        # -------- metric group title --------
        center = x.mean() + bar_width / 2
        if metric == "rc":
            title = "BSC"
        else:
            title = metric.capitalize()
        ax.text(center, 105, title, ha="center", va="bottom", fontsize=25)

        # update base_x (the starting point of the next metric)
        base_x += num_datasets + gap
        if metric_idx < len(metrics) - 1:
            ax.axvline(x=base_x - gap-0.08, color='black', linestyle='-', linewidth=1)

    # -------- axes settings --------
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=25)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=25)

    ax.set_xlabel("Dataset", fontsize=25)
    ax.set_ylabel("Metrics (%)", fontsize=25)

    ax.grid(axis="both", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    left = -bar_width*2
    right = base_x - gap

    ax.set_xlim(left, right)

    leg = ax.legend(fontsize=23, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.4),edgecolor='black')
    texts = leg.get_texts()
    for text in texts:
        if text.get_text() == "RAFT (M)":
            text.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig("fig/exp2_2.png", dpi=300)
    plt.savefig("fig/exp2_2.pdf", dpi=300)




if __name__ == "__main__":
    draw_figure_2_merged()
    # metrics = ["precision", "recall", "f1", "rc"]
    # for metric in metrics:
    #     draw_figure_2(metric)
    
    # draw_know()

    # draw_figure()