#!/usr/bin/env python
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", palette="muted", color_codes=True)


def plotProbabilities(y_gt, scores, suffix, xlabel="p(y=1|X)", width=8, aspect=0.5):
    fig = plt.figure(figsize=(width, width*aspect))
    sns.distplot(scores[y_gt == 0], color=(0.55, 0.70, 0.40),  bins=40, kde=True, hist_kws={'alpha':0.8, 'rwidth':0.8}, kde_kws={"y": None, "linewidth": 2})
    sns.distplot(scores[y_gt == 1], color=(1.00, 0.50, 0.00), bins=40, kde=True, hist_kws={'alpha':0.8, 'rwidth':0.8}, kde_kws={"y": None, "linewidth": 2})
    #plt.xlabel(xlabel)
    #plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig("posterior_histogram_{}.png".format(suffix))
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

data = np.load(sys.argv[1])
suffix = sys.argv[2]

y = data['Y']
P = data['P']
mask = (y >= 0)
plotProbabilities(y[mask], P[mask][:,1], suffix, aspect=0.33)
#plotMetric(y[mask], P[mask][:,1])
