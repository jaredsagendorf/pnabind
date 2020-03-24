#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", palette="muted", color_codes=True)

#from geobind.nn.utils import getMetrics


def plotProbabilities(y_gt, scores, xlabel="p(y=1|X)", labels=["unlabled", "positive"]):
    sns.distplot(scores[y_gt == 0], bins=40, kde=True, hist_kws={'alpha':0.6, 'rwidth':0.8}, kde_kws={"label": labels[0], "linewidth": 2})
    sns.distplot(scores[y_gt == 1], bins=40, kde=True, hist_kws={'alpha':0.6, 'rwidth':0.8}, kde_kws={"label": labels[1], "linewidth": 2})
    plt.xlabel(xlabel)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig("posterior_histogram.png")
    plt.clf()

def plotMetric(y_gt, scores):
    thresholds = np.linspace(0, 1, 20)
    M = []
    for t in thresholds:
        y_pr = (scores >= t).astype(np.int32)
        M.append(balanced_accuracy_score(y_gt, y_pr))
    
    plt.plot(thresholds, M)
    plt.xlabel('threshold')
    plt.ylabel('metric')
    plt.savefig("metric.png")
    plt.clf()
    
y = np.load("data_vertex_labels.npy")
P = np.load("data_vertex_probs.npy")
mask = (y >= 0)
plotProbabilities(y[mask], P[mask][:,1])
#plotMetric(y[mask], P[mask][:,1])
