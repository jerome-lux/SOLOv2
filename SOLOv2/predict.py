import os
import json
from pathlib import Path
import datetime
from copy import deepcopy
from PIL import Image
import pandas
import numpy as np
import skimage as sk
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.measure import find_contours, approximate_polygon
import SOLOv2
from scipy.ndimage import distance_transform_edt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
MINAREA = 512
MINSIZE = 8
BGCOLOR = (90, 140, 200)
deltaL = 10


def get_masks(seg_preds, scores, threshold=0.5, weight_by_scores=False):
    """Get labeled masks from segmentation prediction and scores
    threshold: only pixels with value > threshold are considered foreground
    weight_by_scores: \n
    - if True, each binary masks is weighted by their scores and sorted along the first dimension. Final instance are the argmax tensor of this weighted tensor.
    It means that all pixels in a mask have the same value. It seems that it does not work better that the simpler option...
    \n- if False, we just take the argmax of the segmentation score tensor.
    # TODO, maybe weight the segmentation score by the object score ?
    """

    # Sort instance by scores
    if weight_by_scores:
        # Compute binary masks
        binary_masks = np.where(seg_preds >= threshold, 1, 0)
        sorted_scores_inds = tf.argsort(scores, direction="DESCENDING")
        sorted_scores = tf.gather(scores, sorted_scores_inds).numpy()
        sorted_masks = tf.gather(binary_masks, sorted_scores_inds).numpy()
        # weight mask by their respective scores
        sorted_masks = sorted_masks.T * sorted_scores
        sorted_masks = sorted_masks.T
        # add bg slice
        bg_slice = np.zeros((1, binary_masks.shape[1], binary_masks.shape[2]))
        labeled_masks = np.concatenate([bg_slice, sorted_masks], axis=0)
        # Take argmax (e.g. mask swith higher scores, when two masks overlap)
        labeled_masks = np.argmax(labeled_masks, axis=0)

    # Just take the argmax
    else:
        filt_seg = np.where(seg_preds >= threshold, seg_preds, 0)
        bg_slice = np.zeros((1, filt_seg.shape[1], filt_seg.shape[2]))
        labeled_masks = np.concatenate([bg_slice, filt_seg], axis=0)
        labeled_masks = np.argmax(labeled_masks, axis=0)
        # predicted_instances = np.unique(labeled_masks).size - 1

    return labeled_masks


def box_to_coco(boxes):
    cocoboxes = np.zeros_like(boxes)
    cocoboxes[..., 0:2] = boxes[..., 0:2]
    cocoboxes[..., 2:4] = boxes[..., 2:4] - boxes[..., 0:2]
    return cocoboxes


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, level=0.5, tolerance=0, x_offset=0, y_offset=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = find_contours(padded_binary_mask, 0.5, fully_connected="high")
    contours = [c - 1 for c in contours]
    for contour in contours:
        contour = close_contour(contour)
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            print("Polyshape must have at least 2 points. Skipping")
            continue
        contour = np.flip(contour, axis=1)
        contour[..., 0] += y_offset
        contour[..., 1] += x_offset
        seg = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        seg = [0 if i < 0 else i for i in seg]
        polygons.append(seg)

    return polygons


def filter_instances(
    boxes,
    indexes,
    labels,
    masks,
    minsize=4,
    minarea=64,
    bgcolor=BGCOLOR
):
    """This function first delete small instances and then recompute the regionprops of instances **not touching edges** (not in indexes)
    It saves crops and update the coco and data dictionaries
    It returns the RGB image with flat color on instances **not touching** the edges, as well as the updated labeled masks
    (without small instances and instances touching edges)

    inputs:
    image: RGB image
    boxes: predicted boxes [N, ']
    indexes: indexes of boxes not touching the edges [N'<=N]
    labels: box labels [N]
    masks: labeled instance image
    minsize: minsize of boxes
    minarea: minimum area of an instance
    rand_bg: use random color to mask other instances in a crop
    del_bg: keep the background or only the instance pixels
    """

    bgcolor = np.array(bgcolor) / 255.0

    updated_masks = np.zeros_like(masks)
    label_counter = 0

    # Filter small instances (**which are not touching the edges**). Indexes are the indexes of the labels
    for _, i in enumerate(indexes):
        num_objs = 0
        # Extract mask with label label[i]
        if (boxes[i, 2] - boxes[i, 0] < minsize) or (boxes[i, 3] - boxes[i, 1] < minsize):
            continue
        crop_mask = masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]]
        binary_mask = np.where(crop_mask == labels[i], 1.0, 0.0).astype(np.uint8)

        # Remove small objects
        filtered_binary_mask = sk.morphology.remove_small_objects(
            binary_mask.astype(bool), minarea, connectivity=2
        ).astype(int)

        instance_relabel, num_objs = sk.measure.label(filtered_binary_mask, return_num=True)
        instance_relabel = np.where(instance_relabel > 0, label_counter + instance_relabel, 0)

        # update label image [it also ensure that labels are continuous)
        updated_masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]] = np.where(
            crop_mask == labels[i],
            instance_relabel,
            updated_masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]],
        )
        label_counter += num_objs

    # print(f"Found {label_counter} objects after filtering")

    return updated_masks


def predict_series(
    coco,
    imsize,
    output_dir,
    resolution,
    input_dir,
    model=None,
    model_file=None,
    thresholds=(0.5, 0.5, 0.6),
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=64,
    minsize=4,
    subdirs=False,
    save_imgs=True
):
    """Instance segmentation on an image or a series of images.
    A SOLOv2 network is first used to predict the instance masks, downsampling input images if needed.

    Inputs:
    coco: a dict containing info, licence and categories in coco format
    imsize: size of input images
    input_dir: all the images in input_dir will be processed
    output_dir: where to put teh results
    model: tensorflow model. if None, the model_file must be provided
    model_file: path to tensorflow saved model
    resolution: res of the input images
    thresholds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -COCO object (dict -> saved as json)
    -data dict (dict -> saved as csv)

    Creates on disk:
    - A coco file (json)
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images  in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - individual instances in output_dir/crops folder

    """
    # TODO: maybe compute regionprops on low res images and scale up results to speed things up ?

    # Load model config and weights
    tf.config.run_functions_eagerly(True)

    bgcolor = np.array(bgcolor) / 255.0

    if model is None:
        tf.keras.backend.clear_session()
        model_directory = os.path.dirname(model_file)
        with open(os.path.join(model_directory, "config.json"), "r", encoding="utf-8") as configfile:
            config = json.load(configfile)

        # Creating architecture using config file
        modelconf = SOLOv2.Config(**config)

        model = SOLOv2.model.SOLOv2Model(modelconf)
        print(f"Loading model {model_file}...", end="")
        # Loading weights
        model.load_weights(model_file, by_name=False, skip_mismatch=False).expect_partial()
        print("OK")

    # Retrieve images (either in the input dir only or also in all the subdirectiories)
    img_dict = {}

    if not subdirs:
        for entry in os.scandir(input_dir):
            f = entry.name
            if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_dict[f] = os.path.join(input_dir, f)

    else:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                    img_dict[f] = os.path.join(root, f)

    print("Found", len(img_dict), "images in ", input_dir)

    img_counter = -1

    default_kwargs = {
        "score_threshold": thresholds[0],
        "seg_threshold": thresholds[1],
        "nms_threshold": thresholds[2],
        "max_detections": max_detections,
    }

    print("Resolution of original image:", resolution, "pixels/mm")

    coco["info"]["description"] = str(input_dir)
    coco["annotations"] = []
    coco["images"] = []

    OUTPUT_DIR = Path(output_dir)
    CROPS_DIR = OUTPUT_DIR / Path("crops")
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")

    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    data = {
        "baseimg": [],
        "label": [],
        "res": [],
        "class": [],
        "x0": [],
        "x1": [],
        "y0": [],
        "y1": [],
        "area": [],
        "mass": [],
        "filename": [],
        "axis_major_length": [],
        "axis_minor_length": [],
        "feret_diameter_max": [],
        "max_inscribed_radius":[]
    }

    # network input size
    nx, ny = imsize

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    for counter, imgname in enumerate(keys):
        print(
            "Processing image {} size ({}/{})".format(imgname, counter + 1, len(keys)),
            end="",
        )
        impath = img_dict[imgname]
        PILimg = Image.open(impath)
        fullsize_image = np.array(PILimg) / 255.0
        image = tf.image.resize_with_pad(fullsize_image, nx, ny, antialias=True)
        img_counter += 1
        coco["images"].append(
            {
                "file_name": imgname,
                "coco_url": "",
                "height": PILimg.height,
                "width": PILimg.width,
                "date_captured": "",
                "id": img_counter,
            }
        )
        # Keeping track of the image bbox coordinates in the padded resized image
        ratio = np.max(np.array(fullsize_image.shape[:2]) / np.array(image.shape[:2]))
        lengths = np.array(fullsize_image.shape[:2]) / ratio
        mincoords = np.around((image.shape[:2] - lengths) / 2).astype(int)
        maxcoords = mincoords + np.around(lengths).astype(int)

        fullsize_nx, fullsize_ny, _ = fullsize_image.shape

        # Get network predictions
        seg_preds, scores, cls_labels = model(image[tf.newaxis, ...], training=False, **default_kwargs)

        seg_preds = seg_preds[0].numpy()
        scores = scores[0].numpy()
        cls_labels = cls_labels[0].numpy()

        if scores.size <= 0:  # No detection !
            continue

        # Get labeled mask (int)
        labeled_masks = get_masks(
            seg_preds,
            scores,
            threshold=default_kwargs["seg_threshold"],
            weight_by_scores=weight_by_scores,
        )

        # Extract bboxes
        region_properties = regionprops(labeled_masks)
        pred_boxes = np.array([prop["bbox"] for prop in region_properties])
        labels = np.array([prop["label"] for prop in region_properties])
        indexes = np.arange(labels.shape[0])

        if labels.shape[0] <= 0:
            print("...OK. No instance found !")
            continue

        # Remove small instances
        labeled_masks = filter_instances(
            boxes=pred_boxes,
            indexes=indexes,
            labels=labels,
            masks=labeled_masks,
            minarea=minarea,
            minsize=minsize,
        )
        #in current SOLOv2 implementation, labeled_mask is smaller than image by a factor 2 - so resize it to image shape before cropping
        resized_mask = sk.transform.resize(labeled_masks, (image.shape[0], image.shape[1]), order=0, anti_aliasing=False)
        resized_mask = resized_mask[mincoords[0] : maxcoords[0], mincoords[1] : maxcoords[1]]

        # labeled_masks = labeled_masks[mincoords[0] : maxcoords[0], mincoords[1] : maxcoords[1]]

        if save_imgs:
            vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
            bd = sk.segmentation.find_boundaries(resized_mask, connectivity=2, mode="inner", background=0)
            vizu = label2rgb(resized_mask, image[mincoords[0] : maxcoords[0], mincoords[1] : maxcoords[1]], alpha=0.25, colors=SOLOv2.visualization._COLORS)
            vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
            vizu = np.around(255 * vizu).astype(np.uint8)
            Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        # Resize boxes and mask and recompute region props [crop the image then upscale it to the full-size image shape]

        fullsize_labeled_masks = sk.transform.resize(
            resized_mask,
            (fullsize_nx, fullsize_ny),
            order=0,
            anti_aliasing=False,
        )

        region_properties = regionprops(fullsize_labeled_masks, extra_properties=(max_inscribed_radius, ))
        boxes = np.array([prop["bbox"] for prop in region_properties])
        labels = [prop["label"] for prop in region_properties]
        data["area"].extend([prop["area"] for prop in region_properties])
        data["axis_major_length"].extend([prop["axis_major_length"] for prop in region_properties])
        data["axis_minor_length"].extend([prop["axis_minor_length"] for prop in region_properties])
        data["feret_diameter_max"].extend([prop["feret_diameter_max"] for prop in region_properties])
        data["max_inscribed_radius"].extend([prop["max_inscribed_radius"] for prop in region_properties])

        print(f"...OK. Found {len(labels)} instances. ")

        cocoboxes = box_to_coco(boxes)

        # saving data
        for i, label in enumerate(labels):

            crop_mask = fullsize_labeled_masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]]
            crop = fullsize_image[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]]
            binary_mask = np.where(crop_mask == label, 1.0, 0.0).astype(np.uint8)

            data["baseimg"].append(imgname)
            data["label"].append(labels[i])
            data["res"].append(resolution)
            data["x0"].append(boxes[i, 0])
            data["x1"].append(boxes[i, 2])
            data["y0"].append(boxes[i, 1])
            data["y1"].append(boxes[i, 3])

            # maskname = "{}-MASK-{:03d}.jpg".format(os.path.splitext(imgname)[0],label)
            # Image.fromarray(binary_mask).save(os.path.join(MASKS_DIR,maskname), quality=95)
            cropname = "{}-CROP-{:03d}.jpg".format(os.path.splitext(imgname)[0], labels[i])
            data["filename"].append(cropname)

            crop = np.where(
                (crop_mask[..., np.newaxis] == label) | (crop_mask[..., np.newaxis] == 0),
                crop,
                bgcolor)

            Image.fromarray((255 * crop).astype(np.uint8)).save(os.path.join(CROPS_DIR, cropname), quality=95)

            # Create COCO annotation
            polys = binary_mask_to_polygon(
                binary_mask,
                level=0.5,
                tolerance=1,
                x_offset=cocoboxes[i, 0],
                y_offset=cocoboxes[i, 1],
            )

            coco["annotations"].append(
                {
                    "segmentation": polys,
                    "area": int(data["area"][i]),
                    "iscrowd": 0,
                    "image_id": img_counter,
                    "bbox": [int(b) for b in cocoboxes[i]],
                    "category_id": 19,
                    "id": i,
                }
            )

        # Save labels
        labelname = "{}.png".format(os.path.splitext(imgname)[0])
        Image.fromarray(fullsize_labeled_masks.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

    info_filepath = os.path.join(OUTPUT_DIR, "info.json")
    config = {"SEGNET": str(model_file), "DATA_DIR": str(input_dir)}
    with open(info_filepath, "w", encoding="utf-8") as jsonconfig:
        json.dump(config, jsonconfig)

    print("Saving COCO in ", OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "coco_annotations.json"), "w", encoding="utf-8") as jsonfile:
        json.dump(coco, jsonfile)

    # Set class and mass to None and nan, since they have not been predicted yet
    data["class"] = "None"
    data["mass"] = "nan"
    df = pandas.DataFrame().from_dict(data)
    df = df.set_index("filename")
    df.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"), na_rep="nan", header=True)

    # return COCO and data dict
    return coco, data

def max_inscribed_radius(mask):

    return distance_transform_edt(np.pad(mask, 1)).max()
