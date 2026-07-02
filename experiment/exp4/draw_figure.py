import copy
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker


llms = json.load(open(f"result/eval_result.json", "r", encoding="utf-8"))["individual"]

def draw_figure():
    # Box plot: the x-axis is the log LLM size, the y-axis is F1, and each box contains 6 datasets
    # On top of the box plot, draw a line chart where points are the mean of each dataset and the line connects the means
    data_ours, data_llm = {}, {}
    for llm in llms:
        name, params = llm['name'], int(llm['params'][:-1])
        for i in range(1, 7):
            d = json.load(open(f"log/{name}_prf_dataset{i}.json", "r", encoding="utf-8"))
            if params not in data_ours:
                data_ours[params] = []
                data_llm[params] = []
            data_ours[params].append(d[f"dataset{i}"]['f1_testcase'] * 100)
            d = json.load(open(f"log_llm/{name}_prf_dataset{i}.json", "r", encoding="utf-8"))
            data_llm[params].append(d[f"dataset{i}"]['f1_testcase'] * 100)


    # Draw the box plot
    plt.figure(figsize=(7, 3.5))
    colors = ["#0079CA", "#26A400", "#ff0e0e"]
    colors_b = ["#185E8C", "#2B6A18"]
    positions, plot_data1, plot_data2 = [], [], []
    for i, key in enumerate(list(data_ours.keys())):
        positions.append(key)
        plot_data1.append(data_ours[key])
        plot_data2.append(data_llm[key])


    # Draw the line
    plt.plot(positions, [np.mean(data_ours[key]) for key in data_ours.keys()], marker='o', color=colors_b[0], label="RAFT", linewidth=2, markersize=8)
    plt.plot(positions, [np.mean(data_llm[key]) for key in data_llm.keys()], marker='o', color=colors_b[1], label="E2E LLMs", linewidth=2, markersize=8)
    plt.fill_between(positions, [min(data_ours[key]) for key in data_ours.keys()], [max(data_ours[key]) for key in data_ours.keys()], color=colors[0], alpha=0.2)
    plt.fill_between(positions, [min(data_llm[key]) for key in data_llm.keys()], [max(data_llm[key]) for key in data_llm.keys()], color=colors[1], alpha=0.2)

    # Convert x to log scale
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], fontsize=15)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=15)
    plt.xlabel("LLM Parameter Size (B)", fontsize=15)
    plt.ylabel("F1 (%)", fontsize=15)
    
    legend_elements = [
        (Patch(facecolor=colors[1], alpha=0.2), Line2D([0], [0], marker='o', color=colors_b[1], label='E2E LLMs', linewidth=2)),
        (Line2D([0], [0], marker='o', color=colors_b[0], label='RAFT', linewidth=2), Patch(facecolor=colors[0], alpha=0.2))
    ]
    leg = plt.legend(
        legend_elements,
        ["E2E LLMs", "RAFT"],
        handler_map={tuple: HandlerTuple()}, fontsize=15, edgecolor='black'
    )
    # plt.legend(fontsize=20, edgecolor='black')
    texts = leg.get_texts()
    for text in texts:
        if text.get_text() == "RAFT":
            text.set_fontweight('bold')


    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("fig/exp4.png", dpi=300)
    plt.savefig("fig/exp4.pdf", dpi=300)




if __name__ == "__main__":
    draw_figure()