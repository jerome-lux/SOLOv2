import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from itertools import cycle
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage.transform import resize

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


def draw_instances(image, masks, cls_ids, cls_scores=None, class_ids_to_name=None, show=True, fontscale=5, alpha=0.5, thickness=1):

    """draw masks with class labels and probabilities
    inputs:
    image [H, W, 3]: input image to draw boxes onto
    masks [N, H/2, W/2] : one hot masks without bg
    cls_ids [N]: tensor of class ids
    cls_probs [N]: tensor of class scores
    class_ids_to_name: a dict mapping cls_ids to their names
    show: plot the image
    fontscale : for class names
    alpha: control the labels transparency
    returns:
        annotated image

    """

    nx, ny = masks.shape[1:]

    # bg_slice = np.zeros((1, nx, ny))
    # labeled_masks = np.concatenate([bg_slice, masks], axis=0)
    # labeled_masks = np.argmax(labeled_masks, axis=0)

    output_img = np.array(img_as_float(image))
    output_img = resize(output_img, (nx, ny, 3), order=3, anti_aliasing=True)
    text_img = np.zeros_like(output_img)

    if cls_scores is None:
        cls_scores = np.ones(cls_ids.size)

    colors = cycle(list(_COLORS))

    for i, (class_id, class_score) in enumerate(zip(cls_ids, cls_scores)):

        current_color = next(colors).tolist()

        coords = np.where(masks[i] > 0)
        output_img[coords] = alpha * np.array(current_color) + (1. - alpha) * output_img[coords]

        if class_ids_to_name is not None and fontscale > 0:

            ymin = coords[0].min()
            xmin = coords[1].min()

            class_name = class_ids_to_name[class_id]
            classtext = "{}:{:.0f}%".format(class_name, class_score * 100)
            (text_width, text_height), baseline = cv2.getTextSize(classtext, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            ymin_txt = ymin - baseline if ymin - baseline - text_height > 0 else ymin + text_height
            ymin_bg = ymin - text_height - baseline if ymin - text_height-baseline > 0 else ymin + text_height + baseline
            cv2.rectangle(
                    text_img, (xmin, ymin_bg),
                    (xmin + text_width, ymin),
                    current_color, thickness=cv2.FILLED)
            cv2.putText(text_img, classtext, org=(xmin, ymin_txt), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=fontscale, color=(0.05, 0.05, 0.5), thickness=thickness)

    output_img = np.where(text_img > np.zeros(3), text_img, output_img)

    if show:
        plt.imshow(output_img)
        plt.show()
        return output_img
    else:
        return np.round(output_img*255).astype(np.uint8)



