import numpy as np
import torch
from matplotlib import pyplot as plt
import copy

class_count: int = 3

class_name_to_idx = {
    'background': 0,
    'sidelobe': 1,
    'source': 2,
    'galaxy': 3,
}

def get_missing_preds(gt_boxes, pred_boxes):
    no_pred = []
    for img in gt_boxes:
        if img not in pred_boxes:
            print(f'Image {img} not in predicted images, skipping')
            no_pred.append(img)

    return no_pred

def get_iou(gt_box, pred_box) -> float:
    gt_box_x1, gt_box_y1, gt_box_x2, gt_box_y2 = gt_box
    pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2 = pred_box
    
    x_left = max(gt_box_x1, pred_box_x1)
    x_right = min(gt_box_x2, pred_box_x2)
    y_top = max(gt_box_y1, pred_box_y1)
    y_bottom = min(gt_box_y2, pred_box_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area: float = (x_right - x_left) * (y_bottom - y_top)
        
    # compute the area of both AABBs
    gt_box_area: float = (gt_box_x2 - gt_box_x1) * (gt_box_y2 - gt_box_y1)
    pred_box_area: float = (pred_box_x2 - pred_box_x1) * (pred_box_y2 - pred_box_y1)
        
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou: float = intersection_area / float(gt_box_area + pred_box_area - intersection_area + 1e-8)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

def compute_tp(gt_boxes, pred_boxes, iou_thr=0.6, class_count=3):
    
    true: np.ndarray = np.zeros((1, class_count+1))
    true_positives: np.ndarray = np.zeros((1, class_count+1))
    confusion_matrix: np.ndarray = np.zeros((class_count+1, class_count+1))

        
    # Loop over all of the images
    for i, gt_box_key in enumerate(gt_boxes.keys()):
        if gt_box_key not in pred_boxes.keys():
            c = 2
        assert gt_box_key in pred_boxes.keys()
        
        # Loop over all of the ground truth objects for the current image
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes[gt_box_key]['boxes'], gt_boxes[gt_box_key]['labels'])):     
            # increment true count for the object
            gt_label_idx = class_name_to_idx[gt_label.lower()]
            true[0][gt_label_idx] += 1
            
            # Find associations between true and detected objects according to largest IOU
            index_best: int = -1
            iou_best: float = 0
            score_best: float = 0
            
            # iterate over every detected object for the current image
            for j, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes[gt_box_key]['boxes'], pred_boxes[gt_box_key]['labels'], pred_boxes[gt_box_key]['scores'])):
                # calculate the IoU of the ground truth and detection bounding boxes
                if isinstance(gt_box[0], list):
                    gt_box = gt_box[0]
                if isinstance(pred_box[0], list):
                    pred_box = pred_box[0]
                iou: float = get_iou(gt_box, pred_box)

                if iou > iou_thr and iou > iou_best: # and score threshold?
                    index_best = j
                    iou_best = iou
                    score_best = pred_score
                
            # if a match was found
            if index_best != -1:
                det_label = pred_boxes[gt_box_key]['labels'][index_best]
                det_label_idx = class_name_to_idx[det_label.lower()]
                
                # confusion matrix (for True Negatives)
                confusion_matrix[gt_label_idx][det_label_idx] += 1
                
                # for True Positives
                # if the detection's label is correct (matches that of the ground truth)
                if gt_label == det_label:
                    # increment the TP count for that label/class
                    true_positives[0][gt_label_idx] += 1
    return true, true_positives, confusion_matrix

def compute_pos(pred_boxes):
    pos: np.ndarray = np.zeros((1, class_count+1))
    for pred_box_key in pred_boxes.keys():
        for pred_label in pred_boxes[pred_box_key]['labels']:
            # increment pos count for the detection
            pred_label_idx = class_name_to_idx[pred_label.lower()]
            pos[0][pred_label_idx] += 1
    return pos
                
# The model doesn't make negative predictions, so it doesn't make much sense to include a TN count.
def compute_true_negatives(confusion_matrix):
    true_negatives: np.ndarray = np.zeros((1, class_count+1))
    for i in range(1, class_count+1):
        confusion_matrix_class = copy.deepcopy(confusion_matrix)
        confusion_matrix_class = np.delete(confusion_matrix_class, i, 0)
        confusion_matrix_class = np.delete(confusion_matrix_class, i, 1)
        true_negative_class = np.sum(confusion_matrix_class)
        true_negatives[0][i] = true_negative_class
    return true_negatives

def compute_reliability(true_positives, pos):
    reliability: np.ndarray = np.zeros((1, class_count+1))
    for i, (true_positive_class, pos_class) in enumerate(zip(true_positives[0], pos[0])):
        if pos_class == 0:
            continue
        reliability_class: float = true_positive_class / pos_class
        reliability[0][i] = reliability_class
    return reliability

def compute_completeness(true_positives, true):
    completeness: np.ndarray = np.zeros((1, class_count+1))
    for i, (true_positive_class, true_class) in enumerate(zip(true_positives[0], true[0])):
        if true_class == 0:
            continue
        completeness_class: float = true_positive_class / true_class
        completeness[0][i] = completeness_class
    return completeness

def compute_f1_score(reliability, completeness):
    f1_score: np.ndarray = np.zeros((1, class_count+1))
    for i, (reliability_class, completeness_class) in enumerate(zip(reliability[0], completeness[0])):
        if (reliability_class + completeness_class) == 0:
            continue
        f1_score_class: float = 2 * ((reliability_class * completeness_class) / (reliability_class + completeness_class))
        f1_score[0][i] = f1_score_class
    return f1_score

def log_image(pil_img, labels, boxes, confidence, title, CLASSES, COLORS):

    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.tolist()

    try:
        for cl, (xmin, ymin, xmax, ymax), cs, in zip(labels, boxes, confidence):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl], linewidth=3))
            text = f'{CLASSES[cl]}: {cs:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    except Exception as e:
        print(e)
    finally:
        plt.axis('off')
        return fig
        
def log_gt(pil_img, labels, boxes, title, CLASSES, COLORS):

    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()

    try:
        for cl, (xmin, ymin, xmax, ymax) in zip(labels, boxes):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=COLORS[cl], linewidth=3))
            text = f'{CLASSES[cl]}: GT'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='red', alpha=0.5))
    except Exception as e:
        print(e)
    finally:
        plt.axis('off')
        return fig
