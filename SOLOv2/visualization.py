import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
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
    ]).astype(np.float32).reshape(-1, 3)


def draw_instances(image,
                   labeled_masks,
                   cls_ids,
                   cls_scores=None,
                   class_ids_to_name=None,
                   draw_boundaries=True,
                   show=True,
                   colors=None,
                   fontscale=5,
                   fontcolor=(0,0,0),
                   alpha=0.5,
                   thickness=1):

    import tensorflow as tf
    """draw masks with class labels and probabilities
    inputs:
    image [H, W, 3]: input image to draw boxes onto
    labeled_masks [H/2, W/2] : labeled masks
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class scores
    class_ids_to_name: a dict mapping cls_ids to their names
    show: plot the image
    fontscale : for class names
    alpha: control the labels transparency
    returns:
        annotated image

    """
    if colors is None:
        colors = _COLORS
    else:
        colors = colors

    nx, ny = labeled_masks.shape

    img = tf.image.resize(image, (nx, ny), antialias=True).numpy()

    output_image = label2rgb(labeled_masks,
                             img,
                             bg_label=0,
                             alpha=alpha,
                             colors=colors)

    bd = find_boundaries(labeled_masks, connectivity=2, mode='inner', background=0)
    output_image = np.where(bd[..., np.newaxis],(0, 0, 0), output_image)

    if cls_scores is None:
        cls_scores = np.ones(cls_ids.size)

    # Show class names and scores
    if class_ids_to_name is not None and fontscale > 0:

        colors = cycle(colors)

        for i, (class_id, class_score) in enumerate(zip(cls_ids, cls_scores)):

            current_color = next(colors).tolist()

            coords = tf.where(labeled_masks == i + 1)

            yc = tf.reduce_mean(coords[:, 0]).numpy()
            xc = tf.reduce_mean(coords[:, 1]).numpy()

            class_name = class_ids_to_name[class_id]
            classtext = "{}:{:.0f}%".format(class_name, class_score * 100)
            (text_width,
             text_height), baseline = cv2.getTextSize(classtext,
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      fontscale, thickness)
            ymin_txt = yc - baseline if yc - baseline - text_height > 0 else yc + text_height
            ymin_bg = yc - text_height - baseline if yc - text_height - baseline > 0 else yc + text_height + baseline
            cv2.rectangle(output_image, (xc - text_width//2, ymin_bg),
                          (xc + text_width//2, yc),
                          current_color,
                          thickness=cv2.FILLED)
            cv2.putText(output_image,
                        classtext,
                        org=(xc - text_width//2, ymin_txt),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=fontscale,
                        color=fontcolor,
                        thickness=thickness)

    if show:
        plt.imshow(output_image)
        plt.show()
        return output_image
    else:
        return np.round(output_image * 255).astype(np.uint8)
