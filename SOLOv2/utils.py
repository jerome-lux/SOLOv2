import tensorflow as tf
import numpy as np


@tf.function
def decode_predictions(seg_preds, scores, threshold=0.5, by_scores=True):

    """ Compute the labeled mask array from segmentation predictions.
    If two masks overlap, the one with either the higher score or the higher seg value is chosen
    return labeled array
    inputs:
    seg_preds: (N, H, W) one predicted mask per slice (sigmoid activation)
    scores: (N) score of each predicted instance
    threshold: threshold;to compute binary masks
    by_score (bool): if True, rank the masks by score, else, rank each pixel by their seg_pred value.
    """
    seg_preds = seg_preds.to_tensor()
    binary_masks = tf.where(seg_preds >= threshold, 1, 0)
    nx, ny = tf.shape(seg_preds)[1], tf.shape(seg_preds)[2]

    if by_scores:
        sorted_scores_inds = tf.argsort(scores, direction='DESCENDING')
        sorted_scores = tf.gather(scores, sorted_scores_inds)
        sorted_masks = tf.gather(binary_masks, sorted_scores_inds)
        # weight masks by their respective scores
        sorted_masks = tf.transpose(sorted_masks, [1, 2, 0]) * tf.cast(sorted_scores, tf.float32)
        sorted_masks = tf.transpose(sorted_masks, [2, 0, 1])
        # add bg slice
        bg_slice = tf.zeros((1, nx, ny))
        labeled_masks = tf.concat([bg_slice, binary_masks], axis=0)
        # Take argmax (e.g. mask swith higher scores, when two masks overlap)
        labeled_masks = tf.math.argmax(labeled_masks, axis=0)

    else:
        # Set seg_preds to 0 if < threshold
        filt_seg = tf.where(seg_preds >= threshold, seg_preds, 0.)
        bg_slice = tf.zeros((1, nx, ny))
        labeled_masks = tf.concat([bg_slice, filt_seg], axis=0)
        labeled_masks = tf.math.argmax(labeled_masks, axis=0)

    return labeled_masks


@tf.function
def compute_solo_cls_targets(inputs,
                             shape,
                             strides=[4, 8, 16, 32, 64],
                             grid_sizes=[64, 36, 24, 16, 12],
                             scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
                             mode='diag',
                             offset_factor=0.25):
    """
    inputs:
        (bboxes, labels, classes, masks)
        bboxes: [n, 4] in *normalized coordinates*
        labels: [n] labels of objects
        classes: [n] class ids
        masks: [H/2, W/2]
        shape of original image
        strides: (list of ints) strides of the pyramid levels
        grid_size: size of grids of different FPN levels
        scale_ranges: scale ranges for each level
        mode: either 'min': min(dx, dy) of 'diag': sqrt(dx*dy)
        offset_factor: control the size of the positive box (cx +/- offset_factor*dx*0.5, cy +/- offset_factor*dy*0.5). dx, dy = box side lentghs
        default 0.5 (half-box)

    """
    bboxes = inputs[0]
    labels = inputs[1]
    classes = inputs[2]
    masks = inputs[3]

    # OHE with bg
    maxlabel = tf.reduce_max(labels)
    masks = tf.one_hot(masks, maxlabel + 1)

    nx, ny = shape
    nx = tf.cast(nx, tf.float32)
    ny = tf.cast(ny, tf.float32)
    maxdim = tf.maximum(nx, ny)

    # bboxes size x0, y0, x1, y1 where x1 and y1 are OUTSIDE the box
    dx = tf.maximum((bboxes[..., 2] - bboxes[..., 0]) * (nx - 1) - 1, 0.)
    dy = tf.maximum((bboxes[..., 3] - bboxes[..., 1]) * (ny - 1) - 1, 0.)

    dmin = tf.math.minimum(dx, dy)

    if mode == 'min':
        object_scale = dmin
    elif mode == 'max':
        object_scale = tf.math.maximum(dx, dy)
    else:
        object_scale = tf.math.sqrt(dx * dy)

    idx_per_lvl = []
    bboxes_per_lvl = []
    labels_per_lvl = []
    cls_per_lvl = []

    # Get object ids for each FPN level
    for lvl, (minsize, maxsize) in enumerate(scale_ranges):
        if lvl + 1 < len(grid_sizes) - 1:
            filtered_idx = tf.where(
                (((object_scale >= minsize) & (object_scale <= maxsize) & (dmin >= strides[lvl])) |
                    ((object_scale > maxsize) & (dmin < strides[lvl + 1]) & (dmin >= strides[lvl]))))
        else:
            filtered_idx = tf.where(object_scale >= minsize)

        idx_per_lvl.append(filtered_idx)
        bboxes_per_lvl.append(tf.gather_nd(bboxes, filtered_idx))
        labels_per_lvl.append(tf.gather_nd(labels, filtered_idx))
        cls_per_lvl.append(tf.cast(tf.gather_nd(classes, filtered_idx), tf.int32))

    classes_targets = []
    labels_targets = []

    for lvl, gridsize in enumerate(grid_sizes):

        #Create empty target image
        lvl_imshape = (gridsize * nx//maxdim, gridsize * ny//maxdim)

        cls_img = tf.zeros(lvl_imshape, dtype=tf.int32)
        labels_img = tf.zeros(lvl_imshape, dtype=tf.int32)

        if len(cls_per_lvl[lvl]) > 0:
            # compute  areas of boxes -could use area of masks instead...
            sq_areas = (bboxes_per_lvl[lvl][..., 2]-bboxes_per_lvl[lvl][..., 0]
                        ) * (bboxes_per_lvl[lvl][..., 3]-bboxes_per_lvl[lvl][..., 1])
            # sort by descending areas. if two targets overlap, it is the smallest box that have priority
            ordered_boxes_indices = tf.argsort(sq_areas, axis=-1, direction='DESCENDING')
            # reorder the bboxes tensor and corresponding labels
            # ordered_bboxes_lvl = tf.gather(bboxes_per_lvl[lvl], ordered_boxes_indices)
            ordered_labels = tf.gather(labels_per_lvl[lvl], ordered_boxes_indices)

            # Now generate the targets
            lvl_nx, lvl_ny = lvl_imshape
            lvl_nx = tf.cast(lvl_nx, tf.float32)
            lvl_ny = tf.cast(lvl_ny, tf.float32)
            # Denormed boxes in current levels coordinates
            # x0 = tf.math.maximum(ordered_bboxes_lvl[..., 0] * lvl_nx-1, 0)
            # x1 = tf.math.minimum(ordered_bboxes_lvl[..., 2] * lvl_nx-1, lvl_nx-1)
            # y0 = tf.math.maximum(ordered_bboxes_lvl[..., 1] * lvl_ny-1, 0)
            # y1 = tf.math.minimum(ordered_bboxes_lvl[..., 3] * lvl_ny-1, lvl_ny-1)
            # denorm_boxes_lvl = tf.stack([x0, y0, x1, y1], -1)
            # denorm_boxes_lvl = tf_denormalize_bboxes(ordered_bboxes_lvl, lvl_nx, lvl_ny)

            # Get locations coordinates fo this level [*in current image coordinates !*]
            locations_lvl = tf.cast(compute_locations(1, lvl_imshape), tf.float32)

            # GT targets are located inside the box reduced by the offset_factor
            for i in tf.range(0, tf.shape(cls_per_lvl[lvl])[0]):

                # box = denorm_boxes_lvl[i, ...]
                lab = ordered_labels[i]

                # Normalized coordinates
                coords = tf.where(masks[..., lab] > 0)
                cx = tf.reduce_mean(tf.cast(coords[:, 0], tf.float32)) / tf.cast((tf.shape(masks)[0] - 1), tf.float32)
                cy = tf.reduce_mean(tf.cast(coords[:, 1], tf.float32)) / tf.cast((tf.shape(masks)[1] - 1), tf.float32)

                dx = tf.reduce_max(coords[:, 0]) - tf.reduce_min(coords[:, 0])
                dy = tf.reduce_max(coords[:, 1]) - tf.reduce_min(coords[:, 1])
                dx = tf.cast(dx, tf.float32) / tf.cast((tf.shape(masks)[0] - 1), tf.float32)
                dy = tf.cast(dy, tf.float32) / tf.cast((tf.shape(masks)[1] - 1), tf.float32)

                # inside_indices = tf.where((locations_lvl[:, 0] >= tf.math.floor(cx - x_offset)) &
                #                           (locations_lvl[:, 0] <= tf.math.ceil(cx + x_offset)) &
                #                           (locations_lvl[:, 1] >= tf.math.floor(cy - y_offset)) &
                #                           (locations_lvl[:, 1] <= tf.math.ceil(cy + y_offset)))
                inside_indices = tf.where((locations_lvl[:, 0] >= lvl_nx * (cx - dx * offset_factor) - 0.5) &
                                          (locations_lvl[:, 0] < lvl_nx * (cx + dx * offset_factor) - 0.5) &
                                          (locations_lvl[:, 1] >= lvl_ny * (cy - dy * offset_factor) - 0.5) &
                                          (locations_lvl[:, 1] < lvl_ny * (cy + dy * offset_factor) - 0.5))

                cx = tf.maximum(tf.cast(tf.math.round(lvl_nx * cx - 0.5), tf.int32), 0)
                cy = tf.maximum(tf.cast(tf.math.round(lvl_ny * cy - 0.5), tf.int32), 0)
                center = tf.where((tf.cast(locations_lvl[:, 0], tf.int32) == cx) &
                                 (tf.cast(locations_lvl[:, 1], tf.int32) == cy))
                inside_indices = tf.concat([center, inside_indices], axis=0)

                inside_xc = tf.gather(locations_lvl[:, 0], inside_indices)
                inside_yc = tf.gather(locations_lvl[:, 1], inside_indices)
                inside_xc = tf.reshape(inside_xc, [-1])
                inside_yc = tf.reshape(inside_yc, [-1])
                inside_coords = tf.cast(tf.stack([inside_xc, inside_yc], -1), tf.int32)

                cls_img = tf.tensor_scatter_nd_update(
                    cls_img, inside_coords, tf.zeros(tf.shape(inside_coords)[0], dtype=tf.int32) + cls_per_lvl[lvl][i])
                labels_img = tf.tensor_scatter_nd_update(
                    labels_img, inside_coords, tf.zeros(tf.shape(inside_coords)[0], dtype=tf.int32) + ordered_labels[i])
        # Append the flattened targets

        classes_targets.append(tf.reshape(cls_img, [-1]))
        labels_targets.append(tf.reshape(labels_img, [-1]))

    classes_targets = tf.concat(classes_targets, 0)
    labels_targets = tf.concat(labels_targets, 0)

    return classes_targets, labels_targets


@tf.function
def compute_mask_targets(gt_masks, gt_labels):
    """
    gt_mask is one_hot encoded so that the object with label i is the ith slice
    gt_labels is the flattened (over all FPN levels) vector of labels
    the output is a H x W x npos tensor, with each sloce corresponding to a mask target, aranged in the same order as the class/labels targets
    """

    pos_labels = tf.where(gt_labels > 0)

    for i in pos_labels[:, 0]:
        if i == 0:
            mask_targets = gt_masks[..., gt_labels[i], tf.newaxis]
        mask_targets = tf.concat([mask_targets, gt_masks[..., gt_labels[i], tf.newaxis]], axis=-1)

    return mask_targets


@tf.function
def compute_locations(stride, shape, shift='r'):
    """Compute list of pixels coordinates for a given stride and shape
    if shift == 'r' or 'right, the first point is s//2 else, s//2-1
    """

    if shift.lower() in ['r', 'right']:
        begin = stride // 2
    else:
        begin = stride // 2 - 1

    xc = tf.range(begin, shape[0], stride, dtype=tf.int32)
    yc = tf.range(begin, shape[1], stride, dtype=tf.int32)

    xc, yc = tf.meshgrid(xc, yc)
    xc = tf.reshape(xc, [-1])
    yc = tf.reshape(yc, [-1])
    locations = tf.stack([xc, yc], -1)
    return locations


@tf.function
def tf_normalize_bboxes(bboxes, nx, ny):
    normalized_bboxes = tf.math.divide_no_nan(tf.cast(bboxes, tf.float32), [nx, ny, nx, ny])
    return normalized_bboxes


@tf.function
def tf_denormalize_bboxes(norm_bboxes, nx, ny, rounding='larger'):

    nx = tf.cast(nx, tf.float32)
    ny = tf.cast(ny, tf.float32)
    if rounding == 'larger':
        x0 = tf.math.maximum(tf.math.floor(norm_bboxes[..., 0] * nx), 0)
        x1 = tf.math.minimum(tf.math.ceil(norm_bboxes[..., 2] * nx), nx)
        y0 = tf.math.maximum(tf.math.floor(norm_bboxes[..., 1] * ny), 0)
        y1 = tf.math.minimum(tf.math.ceil(norm_bboxes[..., 3] * ny), ny)
    elif rounding == 'even':
        x0 = tf.math.maximum(tf.math.round(norm_bboxes[..., 0] * nx), 0)
        x1 = tf.math.minimum(tf.math.round(norm_bboxes[..., 2] * nx), nx)
        y0 = tf.math.maximum(tf.math.round(norm_bboxes[..., 1] * ny), 0)
        y1 = tf.math.minimum(tf.math.round(norm_bboxes[..., 3] * ny), ny)
    else:
        x0 = tf.math.maximum(norm_bboxes[..., 0] * nx, 0)
        x1 = tf.math.minimum(norm_bboxes[..., 2] * nx, nx)
        y0 = tf.math.maximum(norm_bboxes[..., 1] * ny, 0)
        y1 = tf.math.minimum(norm_bboxes[..., 3] * ny, ny)

    bboxes = tf.stack([x0, y0, x1, y1], -1)
    # bboxes = tf.cast(bboxes, tf.int32)
    return bboxes


def normalize_bboxes(bboxes, nx, ny):

    normalized_bboxes = bboxes.astype(np.float32)
    normalized_bboxes[..., 0] /= nx
    normalized_bboxes[..., 1] /= ny
    normalized_bboxes[..., 2] /= nx
    normalized_bboxes[..., 3] /= ny
    return normalized_bboxes


def denormalize_bboxes(norm_bboxes, nx, ny):

    bboxes = np.zeros(norm_bboxes.shape).astype(np.int32)
    bboxes[..., 0] = np.maximum(np.around(norm_bboxes[..., 0] * nx), 0)
    bboxes[..., 1] = np.maximum(np.around(norm_bboxes[..., 1] * ny), 0)
    bboxes[..., 2] = np.minimum(np.around(norm_bboxes[..., 2] * nx), nx)
    bboxes[..., 3] = np.minimum(np.around(norm_bboxes[..., 3] * ny), ny)
    return bboxes
