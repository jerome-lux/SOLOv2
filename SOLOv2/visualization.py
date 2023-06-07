import cv2
from . import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from itertools import cycle
from skimage.segmentation import find_boundaries

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def draw_instances(image, boxes, cls_ids, cls_probs, mask=None, class_ids_to_name=None,
                   denorm=True, show=True, fontscale=0.5, thickness=1, alpha=0.5, contours=True, showboxes=True):
    """draw masks and bounding boxes with class probabilities
    image [H,W,3]: input image to draw boxes onto
    boxes: [N,4] tensor of bounding boxes in normalized coordinates
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class probabilities
    mask [H,W], int: if not none, draw the mask of each instance. Each label i corresponds to a box[i-1]
    return:
        annotated image
    show: plot image
    """
    if image.ndim == 2:
        # grey level image
        nx, ny = image.shape
    elif image.ndim == 3:
        # RGB img
        nx, ny, _ = image.shape

#     if not np.issubdtype(image.dtype, np.integer):
#         output_img = np.copy(image)
#     else:
#         output_img = (255*np.copy(image)).astype(np.uint8)
    output_img = np.copy(image)
    if denorm:
        boxes = utils.denormalize_bboxes(boxes, nx, ny).astype(np.int32)

    if cls_probs is None:
        cls_probs = np.ones(cls_ids.size)

#     colors = (255*_COLORS).astype(np.uint8)

    colors = cycle(list(_COLORS))

    i = 0

    for box, class_index, class_score in zip(boxes, cls_ids, cls_probs):

        i += 1

        current_color = next(colors).tolist()

        ymin, xmin, ymax, xmax = box[:4].astype(np.int32)
        ymax += 1
        xmax += 1

        if class_ids_to_name is not None:
            class_name = class_ids_to_name[class_index]
        else:
            class_name = class_index
        classtext = "{}:{:.0f}%".format(class_name, class_score * 100)

        (text_width, text_height), baseline = cv2.getTextSize(classtext, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)

        ymin_bg = ymin - text_height-baseline if ymin - text_height-baseline > 0 else ymin + text_height+baseline
        ymin_txt = ymin - baseline if ymin - baseline - text_height > 0 else ymin + text_height

        if contours:
            boundaries = find_boundaries(mask, mode='inner')
            alpha = np.maximum(alpha, boundaries[..., np.newaxis])

        # Blend mask with image
        if mask is not None:
            # beware x and y are inverted here
            if contours:
                output_img[ymin: ymax, xmin: xmax, :] = np.where(mask[ymin: ymax, xmin: xmax, np.newaxis] == i,
                                                                 alpha[ymin: ymax, xmin: xmax, :] * np.array(current_color) +
                                                                 (1. - alpha[ymin: ymax, xmin: xmax, :]) *
                                                                 output_img[ymin: ymax, xmin: xmax, :],
                                                                 output_img[ymin: ymax, xmin: xmax, :])
            else:
                output_img[ymin: ymax, xmin: xmax, :] = np.where(mask[ymin: ymax, xmin: xmax, np.newaxis] == i,
                                                                 alpha * np.array(current_color) +
                                                                 (1. - alpha) *
                                                                 output_img[ymin: ymax, xmin: xmax, :],
                                                                 output_img[ymin: ymax, xmin: xmax, :])

        if showboxes:
            cv2.rectangle(output_img, (xmin, ymin), (xmax-1, ymax-1), current_color, thickness)
            cv2.rectangle(
                output_img, (xmin, ymin_bg),
                (xmin + text_width, ymin),
                current_color, thickness=cv2.FILLED)
            cv2.putText(output_img, classtext, org=(xmin, ymin_txt), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=fontscale, color=(0, 0, 0), thickness=thickness)

    if show:
        plt.imshow(output_img)
        plt.show()
        return output_img
    else:
        return np.round(output_img*255).astype(np.uint8)


def plot_boxes(boxes, image, cls_ids, cls_probs=None, denorm=True, class_ids_to_name=None,
               mpl_axes=None, cmap='viridis', fontsize=6, fontcolor='lightgreen', linewidth=1, edgecolor='red'):
    """Draw bounding boxes with predicted class and probability
    boxes: [N,4] tensor of bounding boxes in normalized coordinates
    image: input image to draw boxes onto
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class probabilities
    denorm: bool, denormalize bounding boxes
    class_ids_to_name: dictionnary containing classe names
    mpl_axes: an instance of an matplotlib Axe
    return:
        annotated image
    show: plot image
    """

    if image.ndim == 2:
        # grey level image
        nx, ny = image.shape
    elif image.ndim == 3:
        # RGB img
        nx, ny, _ = image.shape

    if mpl_axes is None:
        fig, ax = plt.subplots()
    else:
        ax = mpl_axes

    if denorm:
        proc_boxes = utils.denormalize_bboxes(boxes, nx, ny)
    else:
        proc_boxes = boxes

    if cls_probs is None:
        cls_probs = np.ones(cls_ids.size)

    ax.imshow(image, cmap=cmap, interpolation='nearest')

    if boxes is not None:
        for box, class_index, class_score in zip(proc_boxes, cls_ids, cls_probs):
            if class_ids_to_name is not None:
                class_name = class_ids_to_name[class_index]
            else:
                class_name = class_index
            classtext = "{}: {:.0f}%".format(class_name, class_score * 100)
            x0, y0, x1, y1 = box[:4]

            rect = mpatches.Rectangle((y0, x0), y1 - y0, x1 - x0,
                                      fill=False, edgecolor=edgecolor, linewidth=linewidth)
            ax.text(y0, x0 - int(0.05*ny), classtext, fontsize=fontsize, color=fontcolor)

            ax.add_patch(rect)


def plot_instances(mask, image, boxes, cls_ids, cls_probs=None, denorm=True, class_ids_to_name=None, plot_boxes=False,
                   mpl_axes=None, alpha=0.6, fontsize=6, fontcolor='lightgreen', linewidth=1, edgecolor='red', mask_cmap="default"):
    """Draw predicted masks onto image, with associated predicted class and probability and (optionally) bounding boxes
    mask: image of labeled objects (int)
    image: input image to draw boxes onto
    boxes: [N,4] tensor of bounding boxes in normalized coordinates
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class probabilities
    denorm: bool, denormalize bounding boxes
    class_ids_to_name: dictionnary containing classe names
    plot_boxes (bool) wether to plot boxes or not
    mpl_axes: an instance of an matplotlib Axe
    return:
        annotated image
    show: plot image
    """

    if image.ndim == 2:
        # grey level image
        nx, ny = image.shape
    elif image.ndim == 3:
        # RGB img
        nx, ny, _ = image.shape

    if mpl_axes is None:
        fig, ax = plt.subplots()
    else:
        ax = mpl_axes

    if denorm:
        proc_boxes = utils.denormalize_bboxes(boxes, nx, ny)
    else:
        proc_boxes = boxes

    if cls_probs is None:
        cls_probs = np.ones(cls_ids.size)

    ax.imshow(image)
    masked_mask = np.ma.masked_equal(mask, 0)

    if mask_cmap == "default":
        cmap = ListedColormap(_COLORS, N=cls_probs.size)
    else:
        cmap = masked_mask

    ax.imshow(masked_mask, cmap=cmap, interpolation='nearest', alpha=alpha)

    if boxes is not None:
        for box, class_index, class_score in zip(proc_boxes, cls_ids, cls_probs):
            if class_ids_to_name is not None:
                class_name = class_ids_to_name[class_index]
            else:
                class_name = class_index
            classtext = "{}: {:.0f}%".format(class_name, class_score * 100)
            x0, y0, x1, y1 = box[:4]

            if plot_boxes:
                rect = mpatches.Rectangle((y0 - 0.5, x0 - 0.5), y1 - y0 + 0.5, x1 - x0 + 0.5,
                                          fill=False, edgecolor=edgecolor, linewidth=linewidth)
                ax.add_patch(rect)

            ax.text(y0, x0 - min(5, int(0.02*ny)), classtext, fontsize=fontsize, color=fontcolor)
