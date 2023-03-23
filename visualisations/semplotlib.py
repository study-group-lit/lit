import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_label_heatmap(data, relative=True):
    data = pd.DataFrame(data, ["Entailment", "Neutral", "Contradiction"], ["Entailment", "Neutral", "Contradiction"])
    sn.set(font_scale=1.4)
    if relative:
        ax = sn.heatmap(data/data.to_numpy().sum(), cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='.2%')
    else:
        ax = sn.heatmap(data, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".0f")
    ax.set(xlabel='Predicted Label', ylabel='True Label')
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)
    plt.show()