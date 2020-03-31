# standard packages
import logging
from os.path import join as ospj

# third party packages
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.transforms import PointPairFeatures, GenerateMeshNormals
from scipy import sparse
from sklearn.preprocessing import StandardScaler

gmn = GenerateMeshNormals()
ppf = PointPairFeatures()

def class_weights(y):
    return y.shape[0]/(2*torch.tensor([(y==0).sum(), (y==1).sum()], dtype=torch.float32))

def processBatch(model, batch):
    if(isinstance(model, DataParallel)):
        mask = torch.cat([data.mask for data in batch]).to(model.device)
        y = torch.cat([data.y for data in batch]).to(model.device)
    else:
        batch = batch.to(model.device)
        mask = batch.mask
        y = batch.y
    
    return batch, y, mask

def loadModel(config, nIn=None, nOut=None, tarfile=None):
    if(nIn is None):
        nIn = config["num_features"]
    if(nOut is None):
        nOut = config["num_classes"]
    
    if(config["model"]["name"] == "Net_Conv_EdgePool"):
        import geobind
        from geobind.nn.models import Net_Conv_EdgePool
        model = Net_Conv_EdgePool(nIn, nOut, **config["model"]["kwargs"])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if(tarfile is not None):
        checkpoint = torch.load(tarfile, map_location=device)
        
        # DELETE IN FUTURE
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k.replace("module.", "") # remove module
            new_state_dict[name] = v
        ####################################################
        
        model.load_state_dict(new_state_dict)
    
    model.device = device
    return model

def chooseThreshold(metrics_dict, thresholds, metric='balanced_accuracy', score='metric_value', criteria='max', beta=2, index=False):
    """ Choose a threshold value which meets the following criteria:
        metrics_dict (dict): a dictionary of metrics where each entry is a numpy array of values
                             corresonding to different thresholds.
        score (string): determine what we are going to evaluate
            metric_value - the metric its self
            F-beta - the F-beta score of the metric and threshold with beta weighting the metric.
                     This is useful if we want to choose higher or lower thresholds while still 
                     preferring a good metric score.
        criteria (string):
            min - minimizes the score
            max - maximizes the score
        beta (float): the beta value to use when score=F-beta
    """
    # choose what we are actually evaluating
    values = metrics_dict[metric]
    if(score == 'F-beta'):
        values = (1+beta**2)*(thresholds*values)/((beta**2)*thresholds + values)
    
    # choose how to evaluate
    if(criteria == 'max'):
        idx = np.argmax(values)
    elif(criteria == 'min'):
        idx = np.argmin(values)
    
    # select the metrics corresponding to the index
    for key in metrics_dict:
        if(isinstance(metrics_dict[key], float)):
            continue
        metrics_dict[key] = float(metrics_dict[key][idx])
    
    if(index):
        return idx
    else:
        return thresholds[idx]

def getMetrics(y_gt, prob, threshold=None, n_samples=25, choose_threshold=False, **kwargs):
    if(isinstance(y_gt, list)):
        y_gt = np.array(y_gt)
    if(isinstance(prob, list)):
        prob = np.array(prob)
    
    if(threshold is None):
        # sample n_samples threshold values
        threshold = np.linspace(0, 1, n_samples+2)[1:-1] # skip 0 and 1 values
    else:
        if(isinstance(threshold, float)):
            threshold = np.array([threshold])
            n_samples = 1
            choose_threshold = False
    
    acc = np.empty(n_samples, dtype=np.float32)
    rec = np.empty(n_samples, dtype=np.float32)
    pre = np.empty(n_samples, dtype=np.float32)
    f1s = np.empty(n_samples, dtype=np.float32)
    iou = np.empty(n_samples, dtype=np.float32)
    for i in range(len(threshold)):
        y_pr = (prob[:, 1] >= threshold[i])#.long()
        
        # balanced accuracy
        acc[i] = balanced_accuracy_score(y_gt, y_pr)
        
        # recall
        rec[i] = recall_score(y_gt, y_pr, average='binary')
        
        # precision
        pre[i] = precision_score(y_gt, y_pr, average='binary', zero_division=0)
        
        # f1 score
        f1s[i] = f1_score(y_gt, y_pr, average='binary')
        
        # mean IOU
        iou[i] = jaccard_score(y_gt, y_pr, average='weighted')
    
    M = dict(balanced_accuracy=acc, recall=rec, precision=pre, mean_iou=iou, f1_score=f1s, threshold=threshold)
    if(choose_threshold):
        chooseThreshold(M, threshold, **kwargs)
    elif(n_samples == 1):
        for key in M:
            M[key] = float(M[key][0])
    
    # aurpr
    pre_vals, rec_vals, _ = precision_recall_curve(y_gt, prob[:,1])
    auprc = auc(rec_vals, pre_vals)
    
    # auroc
    fpr, tpr, _ = roc_curve(y_gt, prob[:,1])
    auroc = auc(fpr, tpr)
    M['auroc'] = auroc
    M['auprc'] = auprc
     
    return M

def evaluateDataset(model, dataloader, mask=True, iterate=False, metrics=False, **kwargs):
    y_gts = []
    probs = []
    metrs = []
    with torch.no_grad():
        for batch in dataloader:
            batch, y, bmask = processBatch(model, batch)
            output = model(batch)
            if(mask):
                y = y[bmask].cpu()
                prob = F.softmax(output[bmask], dim=1).cpu()
            else:
                y = y.cpu()
                prob = F.softmax(output, dim=1).cpu()
            
            y_gts.append(y)
            probs.append(prob)
            if(iterate and metrics):
                metrs.append(getMetrics(y, prob, **kwargs))
    
    # decide what to do with each data item
    if(iterate):
        if(metrics):
            return zip(y_gts, probs, metrs)
        else:
            return zip(y_gts, probs)
    else:
        y_gts = torch.cat(y_gts, axis=0)
        probs = torch.cat(probs, axis=0)
    
        if(metrics):
            return getMetrics(y_gts, probs, **kwargs)
        else:
            return y_gts, probs

def createFormattedStrings(fields, values, pad=2, alignment='<'):
    widths = [max(len(f)+pad, 6) for f in fields]
    header_format = ""
    values_format = ""
    for v, w in zip(values, widths):
        if(isinstance(v, int)):
            t = 'd'
        elif(isinstance(v, float)):
            t = '.3f'
        else:
            t = 's'
        header_format += '{:'+alignment+str(w)+'s}'
        values_format += '{:'+alignment+str(w)+t+'}'
    
    header_str = header_format.format(*fields)
    values_str = values_format.format(*values)
    
    return header_str, values_str, widths

def report(metrics_list, sep_char=' | ', header=True, legend=True, **kwargs):
    header_strs = []
    values_strs = []
    tags = []
    widths = []
    for metrics, tag in metrics_list:
        keys = list(metrics)
        keys.sort()
        values = [metrics[k] for k in keys]
        
        hs, vs, w = createFormattedStrings(keys, values, **kwargs)
        header_strs.append(hs)
        values_strs.append(vs)
        widths.append(sum(w))
        tags.append(tag)
    
    if(header):
        if(legend):
            # legend string
            ls = sep_char.join(["{:^{width}s}".format(t, width=w) for t, w in zip(tags, widths)])
            logging.info(ls)
            
        # header string
        hs = sep_char.join(header_strs)
        logging.info(hs)
    
    fs = sep_char.join(values_strs)
    logging.info(fs)

def computePPEdgeFeatures(data, scale=False, norm_only=False):
    # generate edge features
    if(data.norm is None):
        gmn(data)
    
    if(not norm_only):
        ppf(data)
        
        # scale edge features to lie within [0,1]
        if(scale == "clip"):
            e_mean = data.edge_attr.mean(axis=0)
            e_std = data.edge_attr.std(axis=0)
            e_min = e_mean - 2*e_std
            e_max = e_mean + 2*e_std
            data.edge_attr = torch.clamp((data.edge_attr - e_min)/(e_max - e_min), min=0.0, max=1.0)
        elif(scale == "norm"):
            e_mean = data.edge_attr.mean(axis=0)
            e_std = data.edge_attr.std(axis=0)
            data.edge_attr = (data.edge_attr - e_mean)/e_std

def balanceIndices(y, classes, max_percentage=1.0, shuffle=True):
    # create a balanced index set from a vector of labels
    idxs = []
    for c in classes:
        idxs.append(y == c)
    
    # find maximum number of class labels to keep
    nb = int(max_percentage*min([idx.sum() for idx in idxs]))
    for i in range(len(idxs)):
        # exclude any indices above nb
        idx = np.argwhere(idxs[i]).flatten()
        if(shuffle):
            np.random.shuffle(idx)
        idxs[i][idx[nb:]] = False
    
    idxb = np.array(idxs, dtype=bool).sum(axis=0, dtype=bool) # a balanced index set
    
    return idxb

def loadData(data_dir, file_prefixes, 
        nc=2, scale=True, scaler=None, shuffle=True, balance='balanced', remove_mask=False,
        percentage=1.0, batchsize=1, single_gpu=False, ppf_kwargs=None
    ):
    dataset = []
    for f in file_prefixes:
        X = np.load(ospj(data_dir, "%s_vertex_features.npy" % (f)))
        Y = np.load(ospj(data_dir, "%s_vertex_labels.npy" % (f)))
        A = sparse.load_npz(ospj(data_dir, "%s_adj.npz" % (f)))
        pos = np.load(ospj(data_dir, "%s_vertex_positions.npy" % (f)))
        norm = np.load(ospj(data_dir, "%s_vertex_normals.npy" % (f)))
        faces = np.load(ospj(data_dir, "%s_face_indices.npy" % (f)))
        dataset.append((X, Y, A, pos, norm, faces))
    
    if(scale and scaler is None):
        # Fit a scaler to all the training data
        Xc = np.concatenate([x[0] for x in dataset], axis=0)
        scaler = StandardScaler()
        scaler.fit(Xc)
    
    for i in range(len(dataset)):
        X, Y, A, pos, norm, faces = dataset[i]
        
        if(remove_mask):
            # remove previous masking
            Y[(Y == -1)] = 0
        
        if(balance == 'balanced'):
            idxb = balanceIndices(Y, range(nc), max_percentage=percentage)
        elif(balance == 'non-masked'):
            idxb = (Y >= 0)
        elif(balance == 'all'):
            idxb = (Y == Y)
        else:
            raise ValueError("Unrecognized value for `balance` keyword: {}".format(balance))
        
        if(scale):
            X = torch.tensor(scaler.transform(X), dtype=torch.float32)
        else:
            X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.int64)
        e = torch.tensor([A.row, A.col], dtype=torch.int64)
        
        pos = torch.tensor(pos, dtype=torch.float32)
        norm = torch.tensor(norm, dtype=torch.float32)
        faces = torch.tensor(faces.T, dtype=torch.int64)
        
        data = Data(x=X, y=Y, edge_index=e, pos=pos, norm=norm, face=faces, edge_attr=None)
        data.mask = torch.tensor(idxb, dtype=torch.bool)
        dataset[i] = data
        
        if(ppf_kwargs is not None):
            computePPEdgeFeatures(dataset[i], **ppf_kwargs)
    
    logging.info("Finished loaded data from directory %s, using balance: %s", data_dir, balance)
    if(len(dataset) > 1 and torch.cuda.device_count() > 1 and (not single_gpu)):
        # prepare data for parallelization over multiple GPUs
        D = DataListLoader(dataset, batch_size=batchsize, shuffle=shuffle, pin_memory=True)
        D.num_features = dataset[0].x.shape[1]
        D.num_classes = nc
        return D, scaler
    #elif(len(dataset) > 1):
    else:
        # prepate data for single GPU or CPU 
        D = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, pin_memory=True)
        D.num_features = dataset[0].x.shape[1]
        D.num_classes = nc
        return D, scaler
    #else:
        #dataset.num_features = dataset[0].x.shape[1]
        #dataset.num_classes = nc
        #return dataset, scaler
