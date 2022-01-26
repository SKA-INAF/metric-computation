
from __future__ import absolute_import, division, print_function
import argparse
import json
import time
import torch
from metric_functions import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metric_utils import *
from pathlib import Path
from typing import List
import copy
from PIL import Image

CLASSES = ['No-Object', 'sidelobe', 'source', 'galaxy']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556]]

class_name_to_idx = {
    'background': 0,
    'sidelobe': 1,
    'source': 2,
    'galaxy': 3,
}
          
# ## Visualization setup

sns.set_style('white')
sns.set_context('poster')

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def main(args):
    # ## Loading gt and pred annotations

    with open(args.data_dir / args.gt_json) as infile:
        gt_boxes = json.load(infile)

    with open(args.data_dir / args.pred_json) as infile:
        pred_boxes = json.load(infile)


    # Remove missing common predictions (e.g. no object is above the confidence threshold for an image)
    no_pred = get_missing_preds(gt_boxes, pred_boxes)
    for img in no_pred:
        del gt_boxes[img]

    # Get boxes for each class
    gt_boxes_source, pred_boxes_source = split_boxes_by_class(gt_boxes, pred_boxes, 'source')
    gt_boxes_sidelobe, pred_boxes_sidelobe = split_boxes_by_class(gt_boxes, pred_boxes, 'sidelobe')
    gt_boxes_galaxy, pred_boxes_galaxy = split_boxes_by_class(gt_boxes, pred_boxes, 'galaxy')

    # Run per class mAP calculation for each class
    iou_thr = 0.5
    # calculate source mAP
    data = get_avg_precision_at_iou(gt_boxes_source, pred_boxes_source, iou_thr=iou_thr)
    print('Source avg precision: {:.4f}'.format(data['avg_prec']))
    # calculate sidelobe mAP
    data = get_avg_precision_at_iou(gt_boxes_sidelobe, pred_boxes_sidelobe, iou_thr=iou_thr)
    print('Sidelobe avg precision: {:.4f}'.format(data['avg_prec']))
    # calculate galaxy mAP
    data = get_avg_precision_at_iou(gt_boxes_galaxy, pred_boxes_galaxy, iou_thr=iou_thr)
    print('Galaxy avg precision: {:.4f}'.format(data['avg_prec']))

    # Runs it for one IoU threshold for all predictions
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes.copy(), iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    # ## mAP@0.5:0.95 + Reliability-Completeness (RC) (Precision-Recall (PR)) Curve

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.savefig(args.data_dir / 'PRCurve_50:95')
    print(f'PR Curve saved at {args.data_dir}')

    # ## Plot RC (PR) Curve per class


    # Super-Imposed per Class
    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    classes = ['source', 'sidelobe', 'galaxy']
    for i, classification in enumerate(classes):
        for iou_thr in np.linspace(0.5, 0.6, 2):
            if classification == 'source':
                data = get_avg_precision_at_iou(gt_boxes_source, pred_boxes_source, iou_thr=iou_thr)
                colour = 'blue'
            elif classification == 'sidelobe':
                data = get_avg_precision_at_iou(gt_boxes_sidelobe, pred_boxes_sidelobe, iou_thr=iou_thr)
                colour = 'red'
            elif classification == 'galaxy':
                data = get_avg_precision_at_iou(gt_boxes_galaxy, pred_boxes_galaxy, iou_thr=iou_thr)
                colour = 'green'
            avg_precs.append(data['avg_prec'])
            iou_thrs.append(iou_thr)

            precisions = data['precisions']
            recalls = data['recalls']

            if iou_thr == 0.5:
                line_style = 'solid'
            elif iou_thr == 0.6:
                line_style = 'dashed'
            # ax = plot_pr_curve(precisions, recalls, label=classification+' @ {:.2f}'.format(iou_thr),
            #                    color=COLORS[i*len(classes) + j], ax=ax)
            ax = plot_pr_curve(precisions, recalls, label=classification + ' @ {:.2f}'.format(iou_thr),
                                color=colour, ax=ax, linestyle=line_style)


    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100 * np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.title('Reliability-Completeness Curves per Class')
    plt.xlabel('Completeness')
    plt.ylabel('Reliability')
    plt.legend(loc='lower right', title='Class @ IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.savefig('PRCurve_per_class')
    print(f'PR Curve saved at {args.data_dir}')

    # ## Calculate TP, FP, FN, TN, Reliability (Precision), Completeness (Recall), and F1-Score


    # Adapted from https://github.com/SKA-INAF/mrcnn/blob/master/mrcnn/analyze.py lines 1,200-1,340

    iou_thr: float = 0.6

    # [_, sidelobes, sources, galaxies]

    class_count = len(classes)

    true: np.ndarray = np.zeros((1, class_count+1))
    pos: np.ndarray = np.zeros((1, class_count+1))
        
        
    true_positives: np.ndarray = np.zeros((1, class_count+1))
    false_positives: np.ndarray = np.zeros((1, class_count+1))
    false_negatives: np.ndarray = np.zeros((1, class_count+1))
        

    true, true_positives, confusion_matrix = compute_tp(gt_boxes, pred_boxes, iou_thr=iou_thr, class_count=class_count)


    # Display Results
    # True
    print('Classes:\t\t', CLASSES[1:])
    print('True:\t\t\t', true[:,1:])

    # Pos (Detections) (TP + FP)
    # iterate over every detected object in every image

    pos = compute_pos(pred_boxes)
    print('Positive Detections:\t', pos[:,1:])


    # Confusion Matrix
    print('Confusion Matrix:\n', confusion_matrix[1:,1:])


    # True Positives
    print('True Positives:\t\t', true_positives[:,1:])
                    
    # False Positives:
    false_positives = pos - true_positives
    print('False Positives:\t', false_positives[:,1:])

    # False Negatives:
    false_negatives = true - true_positives
    print('False Negatives:\t', false_negatives[:,1:])

    # True Negatives

    true_negatives = compute_true_negatives(confusion_matrix)
    print('True Negatives:\t\t', true_negatives[:,1:])


    # Reliability

    reliability = compute_reliability(true_positives, pos)
    print('Reliability:\t\t', reliability[:,1:])

    # Completeness

    completeness = compute_completeness(true_positives, true)
    print('Completeness:\t\t', completeness[:,1:])

    # F1-Score
    # f1_score = 2 * ((reliability * completeness) / (reliability + completeness))

    f1_score = compute_f1_score(reliability, completeness)
    print('F1-Score:\t\t', f1_score[:,1:])


    reliability: float = true_positives.sum() / pos.sum()
    completeness: float = true_positives.sum() / true.sum()
    f1_score: float = 2 * ((reliability * completeness) / (reliability + completeness))
    print('Total metrics')
    print(f'Reliability: {reliability:.4f}')
    print(f'Completeness: {completeness:.4f}')
    print(f'F1-Score: {f1_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="sample_jsons", help="Path of gt and pred files")
    parser.add_argument("--gt_json", default="gt_boxes.json", help="GT file name")
    parser.add_argument("--pred_json", default="pred_boxes.json", help="Pred file name")

    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)
    main(args)

