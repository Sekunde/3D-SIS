import matplotlib
import numpy as np
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform as sktf
from lib.utils.config import cfg
import math

DEBUG = False

class DetectionMAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5, ignore_class=[], output_dir=None):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.ignore_class = ignore_class
        self.output_dir = output_dir
        self.reset_accumulators()

    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(self.n_class):
            self.total_accumulators.append(APAccumulator())

    def evaluate_mask(self, pred_box, pred_class, pred_conf, pred_mask, gt_box, gt_class, gt_mask, scene_info=[64, 64, 32]):
        pred_class = pred_class.astype(np.int)
        gt_class = gt_class.astype(np.int)
        pred_size = pred_class.shape[0]
        IoU = None
        pred_mask = self.unmold_mask(mask=pred_mask, bbox=pred_box, scene_info=scene_info)
        gt_mask = self.unmold_mask(mask=gt_mask, bbox=gt_box, scene_info=scene_info)
        IoU = DetectionMAP.compute_IoU_mask(pred_mask, gt_mask, pred_conf)

        # mask irrelevant overlaps
        IoU[IoU < self.overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(self.total_accumulators):
            # true positive
            # TP: predict correctly
            # FP: predict wrongly
            # FN: gt missing
            TP, FP, FN = DetectionMAP.compute_TP_FP_FN(pred_class, gt_class, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)

    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, z1, x2, y2, z2] :     Shape [n_pred, 6]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, z1, x2, y2, z2] :  Shape [n_gt, 6]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """
        assert pred_bb.ndim == 2
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        IoU = DetectionMAP.compute_IoU(pred_bb, gt_bb, pred_conf)
        IoU[IoU < self.overlap_threshold] = 0

        # Final match : 1 prediction per GT
        for i, acc in enumerate(self.total_accumulators):
            TP, FP, FN = DetectionMAP.compute_TP_FP_FN(pred_classes, gt_classes, pred_conf, IoU, i)
            acc.inc_predictions(TP, FP)
            acc.inc_not_predicted(FN)

    @staticmethod
    def compute_IoU(prediction, gt, confidence):
        IoU = DetectionMAP.jaccard(prediction, gt)
        return IoU

    @staticmethod
    def compute_IoU_mask(prediction, gt, confidence):
        IoU = DetectionMAP.jaccard_mask(prediction, gt)
        return IoU

    @staticmethod
    def intersect_area(box_a, box_b):
        """
        Compute the area of intersection between two rectangular bounding box
        Bounding boxes use corner notation : [x1, y1, z1, x2, y2, z2]
        Args:
          box_a: (np.array) bounding boxes, Shape: [A,6].
          box_b: (np.array) bounding boxes, Shape: [B,6].
        Return:
          np.array intersection area, Shape: [A,B].
        """
        resized_A = box_a[:, np.newaxis, :]
        resized_B = box_b[np.newaxis, :, :]
        max_xyz = np.minimum(resized_A[:, :, 3:], resized_B[:, :, 3:])
        min_xyz = np.maximum(resized_A[:, :, :3], resized_B[:, :, :3])

        diff_xy = (max_xyz - min_xyz)
        inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
        return inter[:, :, 0] * inter[:, :, 1] * inter[:,:,2]

    @staticmethod
    def jaccard(box_a, box_b):
        """
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
            box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        # adapt when ther is empty predict box list (box_a)
        if box_a.shape[0] == 0:
            return np.zeros([box_a.shape[0], box_b.shape[1]])

        inter = DetectionMAP.intersect_area(box_a, box_b)
        area_a = ((box_a[:, 3] - box_a[:, 0]) * (box_a[:, 4] - box_a[:, 1]) * (box_a[:, 5] - box_a[:, 2]))
        area_b = ((box_b[:, 3] - box_b[:, 0]) * (box_b[:, 4] - box_b[:, 1]) * (box_b[:, 5] - box_b[:, 2]))
        area_a = area_a[:, np.newaxis]
        area_b = area_b[np.newaxis, :]
        union = area_a + area_b - inter
        return inter / union

    @staticmethod
    def jaccard_mask(mask_a, mask_b):
        """
        Compute the jaccard overlap of two sets of masks.  The jaccard overlap
        is simply the intersection over union of two masks.  Here we operate on
        ground truth masks and predict mask.
        E.g.:
            A ∩ B / A ∪ B = sum(A.*B > 0) / sum(A.+B > 0)
        Args:
            mask_a: (np.array) Predicted mask,    Shape: [n_mask, mask_width, mask_height, mask_length]
            mask_b: (np.array) Ground Truth mask, Shape: [n_gt, mask_width, mask_height, mask_length]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = DetectionMAP.intersect_area_mask(mask_a, mask_b)

        iou = np.zeros((mask_a.shape[0], mask_b.shape[0]))
        for i in range(mask_a.shape[0]):
            for j in range(mask_b.shape[0]):
                iou[i,j] = inter[i,j] / sum(sum(sum(mask_a[i] + mask_b[j] > 0)))
        return iou

    @staticmethod
    def intersect_area_mask(mask_a, mask_b):
        """
        Compute the area of intersection between two pooled size masks
        Args:
           mask_a: (np.array) Predicted mask,    Shape: [n_mask, mask_width, mask_height, mask_length]
           mask_b: (np.array) Ground Truth mask, Shape: [n_gt, mask_width, mask_height, mask_length]
        Return:
          np.array intersection area, Shape: [A,B].
        """
        intersect = np.zeros((mask_a.shape[0], mask_b.shape[0]))
        for i in range(mask_a.shape[0]):
            for j in range(mask_b.shape[0]):
                intersect[i,j] = sum(sum(sum((mask_a[i] * mask_b[j] > 0))))
        return intersect

    @staticmethod
    def compute_TP_FP_FN(pred_cls, gt_cls, pred_conf, IoU, class_index):
        if pred_cls.shape[0] == 0:
            return [], [], sum(gt_cls == class_index)

        IoU_mask = IoU != 0

        if pred_cls[0] == -1:
            IoU_mask = IoU_mask
        else:
            IoU_mask = IoU_mask[pred_cls == class_index, :]

        IoU_mask = IoU_mask[:, gt_cls == class_index]

        if pred_cls[0] == -1:
        # IoU number for multiple gt on one pred
            IoU = IoU
        else:
            IoU = IoU[pred_cls == class_index,:]

        IoU = IoU[:, gt_cls == class_index]

        # sum all gt with prediction of this class
        TP = []
        FP = []
        FN = sum(gt_cls == class_index)

        if pred_cls[0] == -1:
            sort_conf_arg = np.argsort(pred_conf[:])[::-1]
        else:
            sort_conf_arg = np.argsort(pred_conf[pred_cls == class_index])[::-1]

        for i in sort_conf_arg:
            ind = -1
            max_overlapping = -1
            for j in range(IoU_mask.shape[1]):
                if IoU_mask[i,j] == True and IoU[i,j] > max_overlapping:
                    ind = j
                    max_overlapping = IoU[i, j]
            if ind != -1:
                TP.append(pred_conf[i])
                IoU_mask[:, ind] = False
                FN -= 1
            else:
                FP.append(pred_conf[i])
        return TP, FP, FN


    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        acc = self.total_accumulators[class_index]
        for i in self.pr_scale:
            precision, recall = acc.precision_recall(i)
            precisions.append(precision)
            recalls.append(recall)

        precisions = precisions[::-1]
        recalls = recalls[::-1]
        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def mAP(self):
        mean_average_precision = []
        for i in range(self.n_class):
            if i in self.ignore_class:
                continue
            precisions, recalls = self.compute_precision_recall_(i, True)
            average_precision = self.compute_ap(precisions, recalls)
            mean_average_precision.append(average_precision)

        if len(mean_average_precision) == 0:
            return 0
        else:
            return sum(mean_average_precision)/len(mean_average_precision)

    def AP(self, idx):
        """

        :param idx:
        :return:
        """
        precisions, recalls = self.compute_precision_recall_(idx, True)
        average_precision = self.compute_ap(precisions, recalls)
        return average_precision

    def finalize(self):
        for acc_ind, acc in enumerate(self.total_accumulators):
            acc.ranking()
            if acc.if_ignore() and acc_ind not in self.ignore_class:
                self.ignore_class.append(acc_ind)

    def unmold_mask(self, mask, bbox, scene_info):
        """

        :param mask:
        :param bbox:
        :param scene_info:
        :return:
        """
        num_mask = len(mask)
        full_mask = np.zeros(shape=(num_mask, *scene_info[:3]), dtype=np.uint8)
        for i in range(num_mask):
            x1, y1, z1, x2, y2, z2 = bbox[i]
            # Put the mask in the right location.
            full_mask[i, int(round(x1)):int(round(x2)), int(round(y1)):int(round(y2)), int(round(z1)): int(round(z2))] = mask[i]
        return full_mask

class APAccumulator:

    """
        Simple accumulator class that keeps track of True positive, False positive and False negative
        to compute precision and recall of a certain class

        predition can only be true positive and false postive (both should have conf)
    """

    def __init__(self):
        # tp: 1, fp: 0
        self.predictions = []
        self.FN = 0
        self.TP = 0

    def inc_predictions(self, TP, FP):
        for tp in TP:
            self.predictions.append([tp, 1.0])
            self.TP += 1
        for fp in FP:
            self.predictions.append([fp, 0.0])

    def inc_not_predicted(self, value=1):
        self.FN += value

    def ranking(self):
        if len(self.predictions) != 0:
            self.predictions = np.stack(self.predictions, 0)
            argsort = np.argsort(self.predictions[:,0])[::-1]
            self.predictions = self.predictions[argsort]
        else:
            self.predictions = np.empty(shape=(0,0))

    def if_ignore(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return True
        else:
            return False

    def precision_recall(self, thresh):

        if thresh == 0.0:
            return (0.0, 0.0)

        TP = 0.0
        FP = 0.0
        total_gt = self.TP + self.FN

        for i in range(self.predictions.shape[0]):
            if self.predictions[i][1] == 1.0:
                TP += 1
            else:
                FP += 1

            recall = TP / float(total_gt)
            precision = TP / (TP + FP)
            # if reach recall, return
            if recall >= thresh:
                return precision, recall

        return (0.0, 1.0)

