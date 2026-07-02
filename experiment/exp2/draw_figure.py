import matplotlib.pyplot as plt
import json

import numpy as np

# colors = ["#1663A9", "#4995C6", "#92C2DD", "#D1D1D1"]
colors = ["#4d7cc5", "#f58e38", "#5AB05C", "#B8B8B8"]


# draw a grouped bar chart
def draw_figure(metric):
    # x-axis is the 6 datasets, y-axis is the metric value, each dataset has 4 bars
    datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6']
    llms = ["deepseek", "gpt", "grok"]
    if metric in ["precision", "recall", "f1"]:
        methods = ["ours", "without_test", "without_repr", ""]
    else:
        methods = ["ours", "without_testability", "without_representation", ""]
    
    data = {dataset: {method: 0 for method in methods} for dataset in datasets}
    for dataset in datasets:
        for method in methods:
            for llm in llms:
                if method == "ours":
                    if metric in ["precision", "recall", "f1"]:
                        dataraw = json.load(open(f"log/{method}_{llm}_prf_{dataset}.json", "r", encoding="utf-8"))
                    else:
                        dataraw = json.load(open(f"log/{method}_{llm}_rc_{dataset}.json", "r", encoding="utf-8"))
                elif method == "":
                    if metric in ["precision", "recall", "f1"]:
                        dataraw = json.load(open(f"log/{llm}_prf_{dataset}.json", "r", encoding="utf-8"))
                    else:
                        dataraw = json.load(open(f"log/{llm}_rc_{dataset}.json", "r", encoding="utf-8"))
                else:
                    if metric in ["precision", "recall", "f1"]:
                        dataraw = json.load(open(f"log/{llm}_{method}_prf_{dataset}.json", "r", encoding="utf-8"))
                    else:
                        dataraw = json.load(open(f"log/{llm}_{method}_rc_{dataset}.json", "r", encoding="utf-8"))
                if metric in ["precision", "recall", "f1"]:
                    data[dataset][method] += dataraw[dataset][metric + "_testcase"] / len(llms)
                else:
                    data[dataset][method] += dataraw[dataset] / len(llms)
    
    for dataset in datasets:
        for method in methods:
            data[dataset][method] *= 100  # convert to percentage

    
    # plotting parameters
    x = np.arange(len(datasets))              # dataset positions
    bar_width = 0.22                           # bar width

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # draw each method
    # labels = ["Full Knowledge (M+R+T)", "Representation Only (M+R)", "Testability Only (T)", "No Knowledge"]
    labels = ["M+R+T", "M+R", "T", "No Know."]
    for i, method in enumerate(methods):
        values = [data[ds][method] for ds in datasets]
        ax.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=labels[i],
            color=colors[i],
            # edgecolor='black',
        )

    # axes and labels
    ax.set_xlabel("Dataset", fontsize=35)
    if metric == "rc":
        metric = "BSC"
    else:
        metric = metric.capitalize()
    # ax.set_ylabel(f"{metric} (%)", fontsize=35)
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels([i for i in range(1, 7)], fontsize=35)
    ax.tick_params(axis='y', labelsize=35)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 100)


    # styling
    ax.grid(axis="both", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"fig/exp2_{metric}.pdf", dpi=300)



def draw_legend():
    labels = ["M+R+T", "M+R", "T", "No Know."]
    fig = plt.figure(figsize=(1, 2))
    ax = fig.add_subplot(111)
    bars = [ax.bar([0], [0], color=c, label=l) for c, l in zip(colors, labels)]
    legend = ax.legend(fontsize=30, ncol=4, loc="center", handletextpad=0.4, # distance from the color patch to the text
        columnspacing=1.5, # distance between columns
        edgecolor='black',
        frameon=True)
    legend.get_frame().set_linewidth(1.0)
    ax.axis('off')
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    plt.savefig("fig/exp2_legend.pdf", dpi=300, bbox_inches=bbox.expanded(1.005, 1.02))


def draw_figure2():
    datasets = [f"dataset{i}" for i in range(1, 7)]
    metrics = ["precision", "recall", "f1", "rc"]
    llms = ["deepseek", "gpt", "grok"]
    methods = ["ours", "without_test", "without_repr", ""]
    labels = ["M+R+T", "M+R", "T", "No Know."]

    num_datasets = len(datasets)
    gap = 0.46
    bar_width = 0.23
    fig, ax = plt.subplots(figsize=(26, 5.5))

    xticks = []
    xtick_labels = []
    base_x = 0

    data = {dataset: {method: {metric: 0 for metric in metrics} for method in methods} for dataset in datasets}
    for dataset in datasets:
        for method in methods:
            for llm in llms:
                for metric in metrics:
                    if method == "ours":
                        dataraw = json.load(open(f"log/{method}_{llm}_{'prf' if metric != 'rc' else 'rc'}_{dataset}.json", "r", encoding="utf-8"))
                    elif method == "":
                        dataraw = json.load(open(f"log/{llm}_{'prf' if metric != 'rc' else 'rc'}_{dataset}.json", "r", encoding="utf-8"))
                    else:
                        if metric == "rc":
                            if method == "without_test":
                                method_file = "without_testability"
                            elif method == "without_repr":
                                method_file = "without_representation"
                            else:
                                method_file = method
                        else:
                            method_file = method
                        dataraw = json.load(open(f"log/{llm}_{method_file}_{'prf' if metric != 'rc' else 'rc'}_{dataset}.json", "r", encoding="utf-8"))
                    if metric in ["precision", "recall", "f1"]:
                        data[dataset][method][metric] += dataraw[dataset][metric + "_testcase"] / len(llms)
                    else:
                        data[dataset][method][metric] += dataraw[dataset] / len(llms)
            for metric in metrics:
                data[dataset][method][metric] *= 100  # convert to percentage
    avg = {method: {metric: 0 for metric in metrics} for method in methods}
    for method in methods:
        for metric in metrics:
            avg[method][metric] = sum(data[ds][method][metric] for ds in datasets) / len(datasets)
    print(json.dumps(avg, indent=4))
    
    for metric_id, metric in enumerate(metrics):
        d = {ds: {m:0 for m in methods} for ds in datasets}
        for dataset in datasets:
            for method in methods:
                d[dataset][method] = data[dataset][method][metric]
        
        x = np.arange(num_datasets) + base_x
        for i, method in enumerate(methods):
            values = [d[ds][method] for ds in datasets]
            ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=labels[i] if metric_id == 0 else None,
                color=colors[i],
            )
        
        xticks.extend(x + bar_width*(len(methods)-1)/2)
        xtick_labels.extend([str(i) for i in range(1, num_datasets + 1)])

        center = x.mean() + bar_width * (len(methods) - 1) / 2
        if metric == "rc":
            metric_name = "BSC"
        else:
            metric_name = metric.capitalize()
        ax.text(center, 105, metric_name, ha='center', va='bottom', fontsize=25)

        base_x += num_datasets + gap
        ax.axvline(x=base_x - gap + 0.07, color='black', linestyle='-', linewidth=1)
            
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=25)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel("Dataset", fontsize=25)
    ax.set_ylabel("Metrics (%)", fontsize=25)
    ax.grid(axis="both", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    left, right = -bar_width/2*3, base_x-gap
    ax.set_xlim(left, right)

    leg = ax.legend(fontsize=23, loc='upper center', ncol=4, bbox_to_anchor=(0.49, 1.4), edgecolor='black')
    texts = leg.get_texts()
    for text in texts:
        if text.get_text() == "M+R+T":
            text.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(f"fig/exp2_all_metrics.png", dpi=300)
    plt.savefig(f"fig/exp2_all_metrics.pdf", dpi=300)






if __name__ == "__main__":
    # metrics = ["precision", "recall", "f1", "rc"]
    # for metric in metrics:
    #     draw_figure(metric)
    # draw_legend()


    draw_figure2()