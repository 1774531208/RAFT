import numpy as np
import matplotlib.pyplot as plt
# before calling fig.legend(), create legend handles without borders
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



def draw_figure():
    # data = {method: {phase: seconds}}
    data = {
        "Experts": {
            # no explicit definition phase
            "generation": (1.486 + 2.042 + 2.814 + 0.936 + 1.725 + 1.414) * 3600 / 2,
            "revise": 0.5 * 3600,
        },
        "E2E LLMs": {
            "generation": (20.616667 * 6 * 60 + 33.45 * 6 * 60 + 29.516667 * 6 * 60) / 3 / 2,
            "revise": 3600 * 5,
        },
        "ACC-Fra": {
            "definition": 60*6 * 60,
            "generation": 18 * 60,
            "revise": 250 * 60,
        },
        "MSG-Spec": {
            "generation": 180 * 60,
            "revise": 228 * 60,
        },
        "LLM4Fin": {
            "definition": 2 * 7 * 24 * 3600,
            "generation": 9.6 * 6,
            "revise": 3600 * 2,
        },
        "RAFT": {
            "definition": 19.6 * 60,
            "generation": 11.5 * 60 * 6,
            "revise": 60 * 60,
        },
        "RAFT+HITL": {
            "definition": 19.6 * 60,
            "generation": 11.5 * 60 * 6,
            "revise": 40 * 60,
        },
    }
    # convert seconds -> minutes
    for method in data:
        for phase in data[method]:
            data[method][phase] /= 60
    labels = list(data.keys())
    x = np.arange(len(labels), dtype=float)
    width = 0.28
    definition_times = [data[l].get("definition", 0) for l in labels]
    generation_times = [data[l].get("generation", 0) for l in labels]
    revise_times = [data[l].get("revise", 0) for l in labels]
    total_times = [definition_times[i] + generation_times[i] + revise_times[i]
                   for i in range(len(labels))]
    has_def = [definition_times[i] > 0 for i in range(len(labels))]
    broken_idx = labels.index("LLM4Fin")  # the only off-scale bar
    c_def, c_gen, c_rev, c_tot = "#369bff", "#ffb066", "#438D42", "#d62728"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]})
    def draw_group(ax, i, legend=False):
        """Draw the (up to) three phase bars + total marker for method i."""
        lab_def = "Exp." if legend else ""
        lab_gen = "Gen." if legend else ""
        lab_rev = "Rev." if legend else ""
        lab_tot = "Total" if legend else ""
        if has_def[i]:
            if i in [2, 4]:
                ax.bar(x[i] - width, definition_times[i], width, color=c_def, label=lab_def, edgecolor="black", linewidth=3, linestyle="--")# dashed border
            else:
                ax.bar(x[i] - width, definition_times[i], width, color=c_def, label=lab_def)
            if i in [0]:
                ax.bar(x[i], generation_times[i], width, color=c_gen, label=lab_gen, edgecolor="black", linewidth=3, linestyle="--")# dashed border
            else:
                ax.bar(x[i], generation_times[i], width, color=c_gen, label=lab_gen)
            ax.bar(x[i] + width, revise_times[i], width, color=c_rev, label=lab_rev, edgecolor="black", linewidth=3, linestyle="--")# dashed border
            xmin, xmax = x[i] - width * 1.5, x[i] + width * 1.5
        else:
            if i in [0]:
                ax.bar(x[i] - width / 2, generation_times[i], width, color=c_gen, label=lab_gen, edgecolor="black", linewidth=3, linestyle="--")# dashed border
            else:
                ax.bar(x[i] - width / 2, generation_times[i], width, color=c_gen, label=lab_gen)
            ax.bar(x[i] + width / 2, revise_times[i], width, color=c_rev, label=lab_rev, edgecolor="black", linewidth=3, linestyle="--")# dashed border
            xmin, xmax = x[i] - width, x[i] + width
        ax.hlines(y=total_times[i], xmin=xmin, xmax=xmax,
                  colors=c_tot, linestyles="-", linewidth=3)
        ax.plot(x[i], total_times[i], color=c_tot, marker="o", markersize=13,
                label=lab_tot, linewidth=3)
    # ---- top subplot: only the off-scale LLM4Fin bar ----
    draw_group(ax1, broken_idx, legend=True)
    ax1.text(x[broken_idx], total_times[broken_idx] + 140, "2 weeks",
             ha="center", fontsize=20, color=c_tot)
    ax1.set_ylim(19900, 20700)
    ax1.set_yticks([20000, 20400])
    ax1.tick_params(axis="y", labelsize=22)
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(axis="x", bottom=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    # ---- bottom subplot: all other methods on a normal scale ----
    for i in range(len(labels)):
        # if i == broken_idx:
        #     continue
        draw_group(ax2, i, legend=False)
        # annotate tiny generation bars (e.g. none here, kept for safety)
        if has_def[i] and generation_times[i] < 10:
            ax2.text(x[i], generation_times[i] + 7, f"~{generation_times[i]:.0f}m",
                     ha="center", fontsize=18, color=c_tot)
    ax2.set_ylabel("Time (m)", fontsize=22)
    ax2.spines["top"].set_visible(False)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=20, rotation=15)
    for tick in ax2.get_xticklabels():
        if tick.get_text().startswith("RAFT"):
            tick.set_fontweight("bold")
    ax2.set_yticks([0, 100, 200, 300, 400, 500, 600])
    ax2.set_ylim(0, 670)
    ax2.tick_params(axis="y", labelsize=22)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    # total labels (in hours) for the on-scale methods
    for i in reversed(range(len(labels))):
        if i == broken_idx:
            continue
        ax2.text(x[i], total_times[i] + 20, f"{total_times[i] / 60:.1f}h",
                 ha="center", va="bottom", fontsize=18, color=c_tot)

    legend_handles = [
        Patch(facecolor=c_def, edgecolor='none', label='Exp.'),
        Patch(facecolor=c_gen, edgecolor='none', label='Gen.'),
        Patch(facecolor=c_rev, edgecolor='none', label='Rev.'),
        Line2D([0], [0], color=c_tot, marker='o', markersize=13, 
           linestyle='-', linewidth=3, label='Total'),
    ]

    # then use these handles instead of getting them from ax1
    fig.legend(handles=legend_handles, labels=['Exp.', 'Gen.', 'Rev.', 'Total'], 
            loc="upper right", fontsize=20, frameon=True, shadow=True, 
            ncol=1, borderpad=0.3, labelspacing=0.2, handletextpad=0.4, 
            bbox_to_anchor=(0.965, 0.95))
    # broken-axis diagonal marks
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none",
                  color="k", mec="k", mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.tight_layout()
    plt.savefig("fig/exp1_time.png", dpi=300, bbox_inches="tight")
    plt.savefig("fig/exp1_time.pdf", dpi=300, bbox_inches="tight")
if __name__ == "__main__":
    draw_figure()