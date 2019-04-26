import torch
from torch.autograd import Variable
import numpy as np

from lib.layer_utils.nms_wrapper import nms
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.utils.config import cfg



def proposal_layer(rpn_cls_prob_level1, rpn_bbox_pred_level1, all_anchors_level1,  
                   rpn_cls_prob_level2, rpn_bbox_pred_level2, all_anchors_level2,
                   rpn_cls_prob_level3, rpn_bbox_pred_level3, all_anchors_level3,
                   scene_info, cfg_key,
                   anchors_filter_level1, anchors_filter_level2, anchors_filter_level3):
    """

    :param rpn_cls_prob <Tensor>: (1, 2, H, W, L, num_anchors)
    :param rpn_bbox_pred <Tensor>: (1, H, W, L, num_anchorsx6), coord. of boxes
    :param scene_info: [64, 32, 64] height, width, length
    :param cfg_key: "TRAIN" or "TEST"
    :param anchors: (NUM_ANCHORSxWxHxL, 6)
    :return: rois in feature map
    """

    # Number of top scoring boxes to keep before apply NMS to RPN proposals
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    # Number of top scoring boxes to keep after applying NMS to RPN proposals
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    # NMS threshold used on RPN proposals
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    # only keep anchors inside the image
    _allowed_border = cfg.ALLOW_BORDER
    if cfg.NUM_ANCHORS_LEVEL1 != 0:
        inds_inside_level1 = np.where(
            (all_anchors_level1[:, 0] >= -_allowed_border) &
            (all_anchors_level1[:, 1] >= -_allowed_border) &
            (all_anchors_level1[:, 2] >= -_allowed_border) &
            (all_anchors_level1[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level1[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level1[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]

        if anchors_filter_level1 is not None:
            if len(anchors_filter_level1) == 0:
                anchors_filter_level1 = [0]
            inds_inside_level1 = inds_inside_level1[anchors_filter_level1]

        anchors_level1 = all_anchors_level1[inds_inside_level1, :]

    if cfg.NUM_ANCHORS_LEVEL2 != 0:
        inds_inside_level2 = np.where(
            (all_anchors_level2[:, 0] >= -_allowed_border) &
            (all_anchors_level2[:, 1] >= -_allowed_border) &
            (all_anchors_level2[:, 2] >= -_allowed_border) &
            (all_anchors_level2[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level2[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level2[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]

        if anchors_filter_level2 is not None:
            if len(anchors_filter_level2) == 0:
                anchors_filter_level2 = [0]
            inds_inside_level2 = inds_inside_level2[anchors_filter_level2]

        anchors_level2 = all_anchors_level2[inds_inside_level2, :]

    if cfg.NUM_ANCHORS_LEVEL3 != 0:
        inds_inside_level3 = np.where(
            (all_anchors_level3[:, 0] >= -_allowed_border) &
            (all_anchors_level3[:, 1] >= -_allowed_border) &
            (all_anchors_level3[:, 2] >= -_allowed_border) &
            (all_anchors_level3[:, 3] < scene_info[0] + _allowed_border) &  #width
            (all_anchors_level3[:, 4] < scene_info[1] + _allowed_border) &  #height
            (all_anchors_level3[:, 5] < scene_info[2] + _allowed_border)   #length
        )[0]

        if anchors_filter_level3 is not None:
            if len(anchors_filter_level3) == 0:
                anchors_filter_level3 = [0]
            inds_inside_level3 = inds_inside_level3[anchors_filter_level3]

        anchors_level3 = all_anchors_level3[inds_inside_level3, :]

    # Get the scores and the bounding boxes
    proposals_batch = []
    scores_batch = []
    levelInds_batch = []
    for i in range(cfg.BATCH_SIZE):
        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            #-------------------------
            # level 1
            #-------------------------
            # (wxhxlxnum_anchors, 6)
            rpn_bbox_pred_reshape_level1 = rpn_bbox_pred_level1[i].view(-1, 6)[inds_inside_level1, :]
             # (wxhxlxnum_anchors)
            scores_level1 = rpn_cls_prob_level1[i, 1, :, :, :, :].view(-1, 1)[inds_inside_level1, :]

            # anchors is in the scene coord
            # return the proposals on scene coord.
            proposals_level1 = bbox_transform_inv(anchors_level1, rpn_bbox_pred_reshape_level1)
            proposals_level1 = clip_boxes(proposals_level1, scene_info[:3])

        if cfg.NUM_ANCHORS_LEVEL2 != 0:
            #-------------------------
            # level 2
            #-------------------------
            # (wxhxlxnum_anchors, 6)
            rpn_bbox_pred_reshape_level2 = rpn_bbox_pred_level2[i].view(-1, 6)[inds_inside_level2, :]
             # (wxhxlxnum_anchors)
            scores_level2 = rpn_cls_prob_level2[i, 1, :, :, :, :].view(-1, 1)[inds_inside_level2, :]

            # anchors is in the scene coord
            # return the proposals on scene coord.
            proposals_level2 = bbox_transform_inv(anchors_level2, rpn_bbox_pred_reshape_level2)
            proposals_level2 = clip_boxes(proposals_level2, scene_info[:3])
            #TODO: eliminate bad box

        if cfg.NUM_ANCHORS_LEVEL3 != 0:
            #-------------------------
            # level 3
            #-------------------------
            # (wxhxlxnum_anchors, 6)
            rpn_bbox_pred_reshape_level3 = rpn_bbox_pred_level3[i].view(-1, 6)[inds_inside_level3, :]
             # (wxhxlxnum_anchors)
            scores_level3 = rpn_cls_prob_level3[i, 1, :, :, :, :].view(-1, 1)[inds_inside_level3, :]

            # anchors is in the scene coord
            # return the proposals on scene coord.
            proposals_level3 = bbox_transform_inv(anchors_level3, rpn_bbox_pred_reshape_level3)
            proposals_level3 = clip_boxes(proposals_level3, scene_info[:3])
            #TODO: eliminate bad box

        #------------------------
        # combine
        #------------------------
        proposals_combined_list = []
        scores_combined_list = []
        levelInds_combined_list = []
        if cfg.NUM_ANCHORS_LEVEL1 != 0:
            proposals_combined_list.append(proposals_level1)
            scores_combined_list.append(scores_level1)
            levelInds_combined_list.append(torch.ones_like(scores_level1))

        if cfg.NUM_ANCHORS_LEVEL2 !=0:
            proposals_combined_list.append(proposals_level2)
            scores_combined_list.append(scores_level2)
            levelInds_combined_list.append(torch.ones_like(scores_level2)*2)

        if cfg.NUM_ANCHORS_LEVEL3 !=0:
            proposals_combined_list.append(proposals_level3)
            scores_combined_list.append(scores_level3)
            levelInds_combined_list.append(torch.ones_like(scores_level3)*3)

        proposals = torch.cat(proposals_combined_list, 0)
        scores = torch.cat(scores_combined_list, 0)[:,0]
        levelInds = torch.cat(levelInds_combined_list, 0)[:,0]

        #proposals = proposals_level2
        #scores = scores_level2[:,0]


        #box= np.stack([np.concatenate([proposals[5222].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5228].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5229].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5319].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5356].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5393].cpu().numpy(), np.ones(1)], 0)], 0)
        #visualize('./vis', 'pos_proposal', data=None, bbox=box)

        #box= np.stack([np.concatenate([proposals[5222].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5223].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5224].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5225].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5226].cpu().numpy(), np.ones(1)], 0),
        #               np.concatenate([proposals[5227].cpu().numpy(), np.ones(1)], 0)], 0)
        #visualize('./vis', 'neg_proposal', data=None, bbox=box)

        # pick up the top region proposals
        scores, order = scores.sort(descending=True)
        #ipdb.set_trace()
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
            scores = scores[:pre_nms_topN].view(-1, 1)
        proposals = proposals[order, :]
        levelInds = levelInds[order]

        # Non-maximal supprression
        keep = nms(proposals, nms_thresh)

        # pick up the top region proposals after NMS
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep,]
        levelInds = levelInds[keep,]

        # support more than 1 scene
        proposals_batch.append(proposals)
        scores_batch.append(scores)
        levelInds_batch.append(levelInds)

    return proposals_batch, scores_batch, levelInds_batch
