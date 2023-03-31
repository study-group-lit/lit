import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_heatmap(data, relative=True, save_path=None):
    data = pd.DataFrame(data, ["Entailment", "Neutral", "Contradiction"], ["Entailment", "Neutral", "Contradiction"])
    sns.set(font_scale=1.4)
    if relative:
        ax = sns.heatmap(data/data.to_numpy().sum(), cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='.2%')
    else:
        ax = sns.heatmap(data, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".0f")
    ax.set(xlabel='Predicted Label', ylabel='True Label')
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)

    if save_path is not None:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches = "tight")
    plt.show()

def plot_metric_heatmap_phenomena(data, models, phenomena, save_path=None):
    data = pd.DataFrame(data, phenomena, models)
    
    sns.set(font_scale=1.4)
    ax = sns.heatmap(data, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".4f")
    ax.set(xlabel='Model', ylabel='Phenomenon')
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)

    if save_path is not None:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches = "tight")
    plt.show()


def plot_metric_heatmap_quantifiers(data, save_path=None):
    quantifiers = ["all", "any", "each", "few", "many", "much", "no", "several", "some", "whole"]
    models = ["Base", "Filtered 3/3\nlonger", "Hypothesis-Only"]
    data = pd.DataFrame(data, quantifiers, models)
    
    sns.set(font_scale=1.4)
    ax = sns.heatmap(data, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt=".4f")
    ax.set(xlabel='Model', ylabel='Quantifier')
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True)

    if save_path is not None:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches = "tight")
    plt.show()