import torch
from torch import nn

class NNRELoss(nn.Module):
    """wrapper of loss function for PU learning"""
    def __init__(self, prior=None, loss=(lambda x, y: torch.sigmoid(-x*y)), gamma=1, beta=0, transform=True):
        super(PULoss,self).__init__()
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss = loss
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1).long()
        self.transform = transform
    
    def forward(self, logits, targets, test=False):
        assert(logits.shape == targets.shape)
        
        # get the positive and unlabeled instances
        if(self.transform):
            targets = self.transformTargets(targets)
        positive, unlabeled = (targets == self.positive, targets == self.unlabeled)
        n_positive = torch.max(self.min_count, torch.sum(positive)).float()
        n_unlabeled = torch.max(self.min_count, torch.sum(unlabeled)).float()
        
        # risk of unlabeled data wrt to negative class 
        R_un = torch.sum(self.loss(logits[unlabeled], -1))/n_unlabeled
        # risk of positive data wrt negative class
        R_pn = torch.sum(self.loss(logits[positive], -1))/n_positive
        # risk of positive data wrt positive class
        R_pp = torch.sum(self.loss(logits[positive], 1))/n_positive
        
        R_p = self.prior*R_pp
        R_n = R_un - self.prior*R_pn
        
        if(R_n >= -self.beta):
            return (R_p + R_n)
        else:
            return -self.gamma*R_n
    
    def transformTargets(self, targets):
        # transform labels to be from [0, 1] to [-1, 1]
        return 2*torch.sign(targets)-1
    
    def estimatePrior(self, data, labels):
        T = TIcE(data, labels)
        self.prior = T.estimate()['alpha']

# Code for TIcE
import numpy as np
import math
from bitarray import bitarray
import time
import heapq

class TIcE(object):
    def __init__(self, data, labels, delta=None, max_bepp=5, maxSplits=500, promis=False, minT=10, nbIts=2):
        # set up
        if(isinstance(data, str)):
            if(data.split('.')[-1] == 'csv'):
                data = np.genfromtxt(data, delimiter=',')
            else:
                data = np.load(data)
        if(isinstance(labels, str)):
            if(labels.split('.')[-1] == 'csv'):
                labels = np.genfromtxt(labels, delimiter=',')
            else:
                labels = np.load(labels)
        labels = bitarray(list(labels==1))
        folds = np.random.randint(5, size=len(data))
        
        self.data = data
        self.labels = labels
        self.folds = folds
        self.delta = delta
        self.max_bepp = max_bepp
        self.maxSplits = maxSplits
        self.promis = promis
        self.minT = minT
        self.nbIts = nbIts
    
    def estimate(self):
        ti = time.time()    
        (c_estimate, c_its_estimates) = tice(self.data, self.labels, self.max_bepp, self.folds, self.delta, 
            nbIterations=self.nbIts,
            maxSplits=self.maxSplits,
            useMostPromisingOnly=self.promis,
            minT=self.minT
        )
        ti = time.time() - ti
    
        alpha = 1.0
        if c_estimate > 0:
            pos = float(self.labels.count()) / c_estimate
            tot = len(self.data)
            alpha = max(0.0, min(1.0, pos / tot))
        
        return {
            "c_its_estimates": c_its_estimates,
            "c_estimate": c_estimate,
            "alpha": alpha,
            "time": ti
        }

def pick_delta(T):
    return max(0.025, 1 / (1 + 0.004 * T))

def low_c(data, label, delta, minT, c=0.5):
    T = float(data.count())
    if T < minT:
        return 0.0
    L = float((data&label).count())
    clow = L/T - math.sqrt(c*(1-c)*(1-delta)/(delta*T))
    return clow

def max_bepp(k):
    def fun(counts):
        return max(list(map(lambda T_P: (0 if T_P[0] == 0 else float(T_P[1]) / (T_P[0] + k)), counts)))
    return fun

def generate_folds(folds):
    for fold in range(max(folds) + 1):
        tree_train = bitarray(list(folds == fold))
        estimate = ~tree_train
        yield (tree_train, estimate)

def tice(data, labels, k, folds, delta=None, nbIterations=2, maxSplits=500, useMostPromisingOnly=False, minT=10,
         n_splits=3):
    if isinstance(labels, np.ndarray):
        labels = bitarray(list(labels == 1))
    
    c_its_ests = []
    c_estimate = 0.5
    
    for it in range(nbIterations):
        c_estimates = []
        global c_cur_best  # global so that it can be used for optimizing queue.
        for (tree_train, estimate) in generate_folds(folds):
            c_cur_best = low_c(estimate, labels, 1.0, minT, c=c_estimate)
            cur_delta = delta if delta else pick_delta(estimate.count())
            
            if useMostPromisingOnly:
                c_tree_best = 0.0
                most_promising = estimate
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels,
                                                                     splitCrit=max_bepp(k), minExamples=minT,
                                                                     maxSplits=maxSplits, c_prior=c_estimate,
                                                                     delta=cur_delta, n_splits=n_splits):
                    tree_est_here = low_c(tree_subset, labels, cur_delta, 1, c=c_estimate)
                    if tree_est_here > c_tree_best:
                        c_tree_best = tree_est_here
                        most_promising = estimate_subset
                        
                c_estimates.append(max(c_cur_best, low_c(most_promising, labels, cur_delta, minT, c=c_estimate)))
            else:
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels,
                                                                     splitCrit=max_bepp(k), minExamples=minT,
                                                                     maxSplits=maxSplits, c_prior=c_estimate,
                                                                     delta=cur_delta, n_splits=n_splits):
                    est_here = low_c(estimate_subset, labels, cur_delta, minT, c=c_estimate)
                    c_cur_best = max(c_cur_best, est_here)
                c_estimates.append(c_cur_best)
        
        c_estimate = sum(c_estimates) / float(len(c_estimates))
        c_its_ests.append(c_estimates)
        
    return c_estimate, c_its_ests

def subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(5), minExamples=10, maxSplits=500,
                     c_prior=0.5, delta=0.0, n_splits=3):
    # This learns a decision tree and updates the label frequency lower bound for every tried split.
    # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
    # The input data is expected to have only binary or continues variables with values between 0 and 1.
    # To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized
    
    # Max: Return all the subsets encountered
    
    all_data = tree_train | estimate
    borders = np.linspace(0, 1, n_splits + 2, True).tolist()[1: -1]
    def makeSubsets(a):
      subsets = []
      options = bitarray(all_data)
      for b in borders:
          X_cond = bitarray(list((data[:, a] < b))) & options
          options &= ~X_cond
          subsets.append(X_cond)
      subsets.append(options)
      return subsets
      
    conditionSets = [makeSubsets(a) for a in range(data.shape[1])]
    
    priorityq = []
    heapq.heappush(priorityq, (-low_c(tree_train, labels, delta, 0, c=c_prior), -(tree_train&labels).count(), tree_train,
                             estimate, set(range(data.shape[1])), 0))
    yield (tree_train, estimate)
    
    n = 0
    minimumLabeled = 1
    while n < maxSplits and len(priorityq) > 0:
        n += 1
        (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = heapq.heappop(priorityq)
        lab_count = -neg_lab_count
        
        best_a = -1
        best_score = -1
        best_subsets_train = []
        best_subsets_estimate = []
        best_lab_counts = []
        uselessAs = set()
        
        for a in available:
          subsets_train = list(map(lambda X_cond: X_cond & subset_train, conditionSets[a]))
          subsets_estimate = list(map(lambda X_cond: X_cond & subset_estimate, conditionSets[a]))  # X_cond & subset_train
          estimate_lab_counts = list(map(lambda subset: (subset & labels).count(), subsets_estimate))
          if max(estimate_lab_counts) < minimumLabeled:
            uselessAs.add(a)
          else:
            score = splitCrit(list(map(lambda subsub: (subsub.count(), (subsub & labels).count()), subsets_train)))
            if score > best_score:
                best_score = score
                best_a = a
                best_subsets_train = subsets_train
                best_subsets_estimate = subsets_estimate
                best_lab_counts = estimate_lab_counts
        
        fake_split = len(list(filter(lambda subset: subset.count() > 0, best_subsets_estimate))) == 1
        
        if best_score > 0 and not fake_split:
          newAvailable = available - {best_a} - uselessAs
          for subsub_train, subsub_estimate in zip(best_subsets_train, best_subsets_estimate):
            yield (subsub_train, subsub_estimate)
          minimumLabeled = c_prior * (1 - c_prior) * (1 - delta) / (delta * (1 - c_cur_best) ** 2)
              
          for (subsub_lab_count, subsub_train, subsub_estimate) in zip(best_lab_counts, best_subsets_train,
                                                                       best_subsets_estimate):
              if subsub_lab_count > minimumLabeled:
                total = subsub_train.count()
                if total > minExamples:  # stop criterion: minimum size for splitting
                  train_lab_count = (subsub_train & labels).count()
                  if lab_count != 0 and lab_count != total:  # stop criterion: purity
                    heapq.heappush(priorityq, (-low_c(subsub_train, labels, delta, 0, c=c_prior), -train_lab_count,
                                               subsub_train, subsub_estimate, newAvailable, depth+1))

def tice_c_to_alpha(c, gamma):
    return 1 - (1 - gamma) * (1 - c) / gamma / c

def tice_wrapper(data, target, k=10, n_folds=10, delta=.2, maxSplits=500, n_splits=40):
    data = min_max_scale(data)
    gamma = target.sum() / target.shape[0]
    c = tice(data, 1 - target, k, np.random.randint(n_folds, size=len(data)),
             delta=delta, maxSplits=maxSplits, minT=10, n_splits=n_splits)[0]
    alpha_tice = tice_c_to_alpha(c, gamma)
    return alpha_tice

def min_max_scale(data):
    data_norm = data - data.min(axis=0)
    data_norm = data_norm / data_norm.max(axis=0)
    return data_norm
