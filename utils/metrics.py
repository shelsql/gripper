import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm

from scipy.spatial import cKDTree
import torch


def transform_pts(pts,tf):
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
        tf = tf[...,None,:,:]
    return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def add_err(pred,gt,model_pts,symetry_tfs=np.eye(4)[None]):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = transform_pts(model_pts, pred)
    gt_pts = transform_pts(model_pts, gt)
    e = np.linalg.norm(pred_pts - gt_pts, axis=-1).mean()
    return e

def adds_err(pred,gt,model_pts):
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    """
    pred_pts = transform_pts(model_pts, pred)
    gt_pts = transform_pts(model_pts, gt)
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e

def compute_auc_sklearn(errs, max_val=0.1, step=0.001):
    from sklearn import metrics
    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val+step, step)
    Y = np.ones(len(X))
    for i,x in enumerate(X):
        y = (errs<=x).sum()/len(errs)
        Y[i] = y
        if y>=1:
            break
    auc = metrics.auc(X, Y) / (max_val*1)
    return auc

def compute_auc_all(preds, gts, model_pts,max=0.1,step=0.001):
    assert(preds.shape[0] == gts.shape[0])
    n = preds.shape[0]
    add_all = []
    adds_all = []
    for i in range(n):
        add = add_err(preds[i], gts[i], model_pts)
        adds = adds_err(preds[i], gts[i], model_pts)
        add_all.append(add)
        adds_all.append(adds)
    add_auc = compute_auc_sklearn(add_all,max,step)
    adds_auc = compute_auc_sklearn(adds_all,max,step)
    metrics = {
        "add_auc": add_auc,
        "adds_auc": adds_auc
    }
    return metrics