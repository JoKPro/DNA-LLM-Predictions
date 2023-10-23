import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_pr_curves(*datasets, figsize=(8, 6), **kwargs):
    """Plot Precision Recall Curve for variable number of datasets.
    Parameters
    ----------
    *datasets: tuple[3]
        Tuple containing (predicion_scores: list, truth_labels: list, name: str)
    
    figsize: tuple[2]
        figsize for plt.figure
        
    **kwargs: 
        other arguments to pass to sns.lineplot
    """
    
    # Create a figure to plot the precision-recall curves
    plt.figure(figsize=(8, 6))
    
    for dataset in datasets:
        scores = dataset[0]
        labels = dataset[1]
        name = dataset[2]
        precision, recall, _ = precision_recall_curve(labels, scores)
        area = auc(recall, precision, )
        sns.lineplot(x=recall, y=precision, label=f"{name} AUC: {area:.2f}", **kwargs)
        
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
        
        
def plot_roc_curves(*datasets, figsize=(8, 6), **kwargs):
    # Create a figure to plot the precision-recall curves
    plt.figure(figsize=(8, 6))
    
    for dataset in datasets:
        scores = dataset[0]
        labels = dataset[1]
        name = dataset[2]
        fpr, tpr, _ = roc_curve(labels, scores)
        area = auc(fpr, tpr)
        sns.lineplot(x=fpr, y=tpr, label=f"{name} AUC: {area:.2f}", **kwargs)   
        
    plt.title("ROC-Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()