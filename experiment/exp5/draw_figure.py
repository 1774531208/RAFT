import numpy as np
import matplotlib.pyplot as plt
import json


def draw_figure(metric):
    # load data
    # data = {method: [values of 6 datasets]}
    data = {
        "Expert": [],
        "TARGET": [],
        "E2E LLM": [],
        "Ours Ind.": [],
        "Ours Agg.": [],
    }

    for i in range(1, 6):
        for method in ['expert', 'target', 'gpt', 'grok', 'deepseek', 'ours_gpt', 'ours_grok', 'ours_deepseek', "ours_gpt_only", "ours_grok_only", "ours_deepseek_only"]:
            if i >= 4 and method == "target":
                    continue
            if metric != "rc":
                d = json.load(open(f'log/{method}_prf_dataset{i}.json', 'r', encoding='utf-8'))
            else:
                d = json.load(open(f'log/{method}_rc_dataset{i}.json', 'r', encoding='utf-8'))
            if method == 'expert':
                if metric != "rc":
                    data["Expert"].append(d[f'dataset{i}'][f'{metric}_testcase'])
                else:
                    data["Expert"].append(d[f'dataset{i}'])
            elif method == 'target':
                if metric != "rc":
                    data["TARGET"].append(d[f'dataset{i}'][f'{metric}_testcase'])
                else:
                    data["TARGET"].append(d[f'dataset{i}'])
            elif method == 'gpt' or method == "grok" or method == "deepseek":
                if metric != "rc":
                    data["E2E LLM"].append(d[f'dataset{i}'][f'{metric}_testcase'])
                else:
                    data["E2E LLM"].append(d[f'dataset{i}'])
            elif method == 'ours_gpt' or method == "ours_grok" or method == "ours_deepseek":
                if metric != "rc":
                    data["Ours Agg."].append(d[f'dataset{i}'][f'{metric}_testcase'])
                else:
                    data["Ours Agg."].append(d[f'dataset{i}'])
            elif method == 'ours_gpt_only' or method == "ours_grok_only" or method == "ours_deepseek_only":
                if metric != "rc":
                    data["Ours Ind."].append(d[f'dataset{i}'][f'{metric}_testcase'])
                else:
                    data["Ours Ind."].append(d[f'dataset{i}'])

    for key in data:
        data[key] = [v * 100 for v in data[key]]  # convert to percentage


    colors = ['#e74c3c', '#3498db', "#2ca02c", '#9467bd', "#ff7f0e"]
    width = 0.8
    plt.figure(figsize=(6, 3.5))

    plot_data = []
    positions = []

    for i, method in enumerate(list(data.keys())):
        plot_data.append(data[method])
        positions.append(0.1 + i)

    # draw boxplot
    bp = plt.boxplot(
        plot_data,
        positions=positions,
        widths=width,
        patch_artist=True,  # allow fill color
        boxprops=dict(color='black'),
        capprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        flierprops=dict(marker='o'),
        medianprops=dict(color='black', linewidth=2)
    )
    
    # set different colors for different methods
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])

    x = [0.1 + i for i in range(5)]
    plt.xticks(x, ['Exp', 'TAR', 'LLM', 'O.I ', 'O.A'], fontsize=25)
    plt.ylim(0, 100)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=25)

    # add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if metric == "precision":
        plt.ylabel("Precision (%)", fontsize=25)
    elif metric == "recall":
        plt.ylabel("Recall (%)", fontsize=25)
    elif metric == "f1":
        plt.ylabel("F1 (%)", fontsize=25)
    elif metric == "rc":
        plt.ylabel("BSC (%)", fontsize=25)
    
    # plt.legend(['Expert', 'LLM4Fin', 'E2E LLM', 'Ours Ind.', 'Ours Agg.'], fontsize=20)


    plt.tight_layout()
    plt.savefig(f'fig/exp3_{metric}.png', dpi=300, bbox_inches='tight')





if __name__ == "__main__":
    draw_figure("precision")
    draw_figure("recall")
    draw_figure("f1")
    draw_figure("rc")