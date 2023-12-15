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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

SAVE_IMGS = True
VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
MINAREA = 2048
MINSIZE = 32
BGCOLOR = (94, 160, 220)
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


def filter_and_save_instances(
    imgname,
    image,
    boxes,
    indexes,
    labels,
    masks,
    coco,
    data,
    img_id,
    crops_dir,
    resolution,
    save_crops=True,
    minsize=MINSIZE,
    minarea=MINAREA,
    rand_bg=False,
    bgcolor=BGCOLOR,
    del_bg=False,
):
    """This function first delete small instances and then recompute the regionprops of instances **not touching edges** (not in indexes)
    It saves crops and update the coco and data dictionaries
    It returns the RGB image with flat color on instances **not touching** the edges, as well as the updated labeled masks
    (without small instances and instances touching edges)

    inputs:
    imgname: filename of RGB image
    image: RGB image
    boxes: predicted boxes [N, ']
    indexes: indexes of boxes not touching the edges [N'<=N]
    labels: box labels [N]
    masks: labeled instance image
    coco: coco dict to update
    data: data dict to update
    img_id: id of the image
    crops_dir: where to save crops
    minsize: minsize of boxes
    minarea: minimum area of an instance
    save_crops: if False the individual instances are not saved
    rand_bg: use random color to mask other instances in a crop
    del_bg: keep the background or only the instance pixels
    """

    instance_counter = 0
    cleaned_image = deepcopy(image)

    bgcolor = np.array(bgcolor) / 255.

    updated_masks = np.zeros_like(masks)
    label_counter = 0
    # Filter small instances (**which are not touching the edges**). Indexes are the indexes of the labels
    for k, i in enumerate(indexes):
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
            masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]] == labels[i],
            instance_relabel,
            updated_masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]],
        )
        label_counter += num_objs
    # print(f"Found {label_counter} objects after filtering")

    # recompute bounding boxes
    region_properties = regionprops(updated_masks)
    boxes = np.array([prop["bbox"] for prop in region_properties])
    labels = np.array([prop["label"] for prop in region_properties])
    areas = np.array([prop["area"] for prop in region_properties])

    cocoboxes = box_to_coco(boxes)

    # print("labels after filtering", labels)

    for i, label in enumerate(labels):
        # print("Saving label", label)

        if rand_bg:
            bgcolor = np.random.random(3)

        crop_mask = updated_masks[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]]
        binary_mask = np.where(crop_mask == label, 1.0, 0.0).astype(np.uint8)

        # On remplace l'instance qui touche le bord par un fond bleu
        cleaned_image[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]] = np.where(
            binary_mask[..., np.newaxis] > 0,
            bgcolor,
            cleaned_image[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]],
        )

        data["baseimg"].append(imgname)
        data["label"].append(labels[i])
        data["res"].append(resolution)
        data["x0"].append(boxes[i, 0])
        data["x1"].append(boxes[i, 2])
        data["y0"].append(boxes[i, 1])
        data["y1"].append(boxes[i, 3])
        data["area"].append(areas[i])

        instance_counter += 1
        # maskname = "{}-MASK-{:03d}.jpg".format(os.path.splitext(imgname)[0],label)
        # Image.fromarray(binary_mask).save(os.path.join(MASKS_DIR,maskname), quality=95)
        cropname = "{}-CROP-{:03d}.jpg".format(os.path.splitext(imgname)[0], labels[i])
        data["filename"].append(cropname)

        if save_crops:
            crop = image[boxes[i, 0] : boxes[i, 2], boxes[i, 1] : boxes[i, 3]]
            if del_bg:
                crop = np.where(crop_mask[..., np.newaxis] == label, crop, bgcolor)
            else:
                crop = np.where(
                    (crop_mask[..., np.newaxis] == label) | (crop_mask[..., np.newaxis] == 0),
                    crop,
                    bgcolor,
                )

            Image.fromarray((255 * crop).astype(np.uint8)).save(os.path.join(crops_dir, cropname), quality=95)

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
                "area": int(areas[i]),
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [int(b) for b in cocoboxes[i]],
                "category_id": 19,
                "id": instance_counter,
            }
        )
    return cleaned_image, updated_masks, labels.size


def predict(
    coco,
    imsize,
    input_dir,
    output_dir,
    model_file,
    resolution,
    deltaL=10,
    thresholds=(0.5, 0.5, 0.6),
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=MINAREA,
    minsize=MINSIZE,
):
    """Instance segmentation on an image or a series of images.
    A SOLOv2 network is first used to preddict the instance masks, downsampling input images if needed.
    Objects straddling two frames are used to determine the two overlap bands.
    An image containing only objects touching the edges is formed from the overlap bands.
    Entire objects are then extracted from this intermediate image.
    Dans le cas où il n'y a qu'une image, les objets touchant le bord inférieur ne sont donc pas traités.

    Inputs:
    coco: a dict containing info, licence and categories in coco format
    imsize: size of input images
    input_dir: all the images in input_dir will be processed
    output_dir: where to put teh results
    model_file: path to tensorflow saved model
    deltaL: if a box is less than deltaL from the image edge, the object is considered to be touching the edge.
    resolution: res of the input images
    trhesolds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -COCO object (dict)
    -data dict

    Creates on disk:
    - A coco file (json)
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images  in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - individual instances in output_dir/crops folder
    - newly created overlap bands images (full resolution) in output_dir/images

    """
    # TODO: maybe compute regionprops on low res images and scale up results to speed things up ?

    # Load model config and weights
    tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    model_directory = os.path.dirname(model_file)
    with open(os.path.join(model_directory, "config.json"), "r", encoding="utf-8") as configfile:
        config = json.load(configfile)

    # Creating architecture using config file
    modelconf = SOLOv2.Config(**config)

    detector = SOLOv2.model.SOLOv2Model(modelconf)
    print(f"Loading model {model_file}...", end="")
    # Loading weights
    detector.load_weights(model_file, by_name=False, skip_mismatch=False).expect_partial()
    print("OK")

    # Retrieve images
    img_dict = {}
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

    coco["info"]["description"] = input_dir
    coco["annotations"] = []
    coco["images"] = []

    OUTPUT_DIR = os.path.join(output_dir, str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    CROPS_DIR = OUTPUT_DIR / Path("crops")
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")
    OVERLAPPING_IMGS_DIR = OUTPUT_DIR / Path("images")

    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(OVERLAPPING_IMGS_DIR, exist_ok=True)

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
    }

    nx, ny = imsize

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    for counter, imgname in enumerate(keys):
        print(
            "Processing image {} size ({}/{})".format(imgname, counter, len(keys)),
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
        # Resize if the ratio nx/ny is not the same (if not, bboxes are not rescaled correctly)
        fullsize_nx, fullsize_ny, _ = fullsize_image.shape
        if nx * fullsize_ny != fullsize_nx * ny:
            ratio = min(nx / fullsize_nx, ny / fullsize_ny)
            fullsize_image = tf.image.resize_with_pad(fullsize_image, int(nx / ratio), int(ny / ratio), antialias=True)
            fullsize_nx, fullsize_ny, _ = fullsize_image.shape

        # Get network predictions
        seg_preds, scores, cls_labels = detector(image[tf.newaxis, ...], training=False, **default_kwargs)

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

        pred_mask_fullsize = sk.transform.resize(
            labeled_masks, (fullsize_nx, fullsize_ny), order=0, anti_aliasing=False
        )

        # Extract bboxes
        region_properties = regionprops(pred_mask_fullsize)
        pred_boxes = np.array([prop["bbox"] for prop in region_properties])
        labels = np.array([prop["label"] for prop in region_properties])

        # middle indexes: indexes of boxes not touching edges,
        # up indexes: boxes touching upper edge (i.e. x->0)
        # TODO: check if an object touches both edges (normalement non !)
        if counter == 0:
            # on extrait toutes les boites sauf celle touchant le "bas" (x=nx)
            middle_indexes = np.where(pred_boxes[:, 2] < fullsize_nx - deltaL)
        elif counter == len(keys) - 1:
            # On extrait toutes les boites sauf celles touchant le "haut" (x=0)
            middle_indexes = np.where(pred_boxes[:, 0] > deltaL)
        else:
            # On extrait toutes les boites sauf celles touchant le "haut" (x=0) ET le bas (x=nx)
            middle_indexes = np.where((pred_boxes[:, 2] < fullsize_nx - deltaL) & (pred_boxes[:, 0] > deltaL))

        middle_labels = []
        if middle_indexes[0].size > 0:
            middle_labels = labels[middle_indexes]

        # le "haut de l'image" correspond à nx=0
        up_indexes = np.where(pred_boxes[:, 0] <= deltaL)
        up_labels = labels[up_indexes]

        # Save crops of instances not touching edges and return image without them
        # also delete small instances and update masks
        if middle_indexes[0].size > 0:
            (
                fullsize_cleaned_image,
                pred_mask_fullsize,
                n_inst,
            ) = filter_and_save_instances(
                imgname=imgname,
                image=fullsize_image,
                boxes=pred_boxes,
                indexes=middle_indexes[0],
                labels=labels,
                masks=pred_mask_fullsize,
                coco=coco,
                data=data,
                img_id=img_counter,
                resolution=resolution,
                crops_dir=CROPS_DIR,
                minarea=minarea,
                minsize=minsize,
                rand_bg=False,
                del_bg=False,
                bgcolor=bgcolor,
                save_crops=True,
            )

        print(
            "...OK. Found {} instances,  [{} objects touching the edges]. ".format(
                n_inst, -middle_indexes[0].size + n_inst
            )
        )

        # Save labels
        labelname = "{}.png".format(os.path.splitext(imgname)[0])
        Image.fromarray(pred_mask_fullsize.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

        if SAVE_IMGS:
            vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
            resized_masks = sk.transform.resize(pred_mask_fullsize, (nx, ny), order=0, anti_aliasing=False)
            bd = sk.segmentation.find_boundaries(resized_masks, connectivity=2, mode="inner", background=0)
            vizu = label2rgb(resized_masks, image, alpha=0.25, colors=SOLOv2.visualization._COLORS)
            vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
            vizu = np.around(255 * vizu).astype(np.uint8)
            Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        # Create overlapping image using previous bottom image and current up image
        if counter > 0:
            # hauteur bande de recouvrement haute et basse
            # ici bottom_boxes correspond aux objets détectée sur l'image précédente
            fullsize_boxes_up = pred_boxes[up_indexes]
            # print("boxes touching upper edge:", fullsize_boxes_up)
            fullsize_overlap_image = np.zeros_like(fullsize_image)

            # S'il n'y a pas d'objets touchant les bords en haut de l'image actuelle ou en bas de la précédente alors on passe
            if not (fullsize_boxes_up.size == 0 and prev_fullsize_bottom_boxes.size == 0):
                # Up correspond au haut de l'image cad coords à partir de 0

                if fullsize_boxes_up.size > 0:
                    lxup = fullsize_boxes_up[:, 2].max()
                else:
                    lxup = 0

                # bottom correspond au "bas" de l'image (vers les coords croissantes)
                if prev_fullsize_bottom_boxes.size > 0:
                    lxdown = prev_fullsize_bottom_boxes[:, 0].min()
                else:
                    lxdown = fullsize_nx

                # print("lxup, lxdown", lxup, lxdown)

                if lxup == 0 and lxdown < fullsize_nx:
                    fullsize_overlap_image = prev_fullsize_cleaned_image[lxdown:, ...]
                    # print("coucou",fullsize_overlap_image.shape )

                elif lxup > 0 and lxdown == fullsize_nx:
                    fullsize_overlap_image = fullsize_cleaned_image[0:lxup, ...]
                else:
                    fullsize_overlap_image = np.concatenate(
                        [
                            prev_fullsize_cleaned_image[lxdown:, ...],
                            fullsize_cleaned_image[0:lxup, ...],
                        ],
                        0,
                    )
                # Resize to default size
                fullsize_overlap_image = tf.image.resize_with_pad(
                    fullsize_overlap_image, fullsize_nx, fullsize_ny, antialias=True
                ).numpy()

                (
                    fullsize_overlap_nx,
                    fullsize_overlap_ny,
                    _,
                ) = fullsize_overlap_image.shape

                overlap_image = tf.image.resize_with_pad(fullsize_overlap_image, nx, ny, antialias=True)

                # Get network predictions
                o_seg_preds, o_scores, o_cls_labels = detector(
                    overlap_image[tf.newaxis, ...], training=False, **default_kwargs
                )

                # Because batchsize=1
                o_seg_preds = o_seg_preds[0].numpy()
                o_scores = o_scores[0].numpy()
                o_cls_labels = o_cls_labels[0].numpy()

                o_labeled_masks = get_masks(
                    o_seg_preds,
                    o_scores,
                    threshold=default_kwargs["seg_threshold"],
                    weight_by_scores=weight_by_scores,
                )

                if o_seg_preds.shape[0] > 0:
                    # Save overlap image
                    o_imgname = "OVERLAP_{}-{}.jpg".format(
                        os.path.splitext(prev_imgname)[0], os.path.splitext(imgname)[0]
                    )
                    print("Processing overlapping image {}...".format(o_imgname), end="")
                    o_PILimg = Image.fromarray(np.around(fullsize_overlap_image * 255).astype(np.uint8))
                    o_PILimg.save(os.path.join(OVERLAPPING_IMGS_DIR, o_imgname), quality=95)
                    img_counter += 1

                    o_pred_mask_fullsize = sk.transform.resize(
                        o_labeled_masks,
                        (fullsize_overlap_nx, fullsize_overlap_ny),
                        order=0,
                        anti_aliasing=False,
                    )

                    # Extract bboxes
                    o_region_properties = regionprops(o_pred_mask_fullsize)
                    o_pred_boxes = np.array([prop["bbox"] for prop in o_region_properties])
                    o_labels = np.array([prop["label"] for prop in o_region_properties])
                    n_inst_before = o_labels.size

                    coco["images"].append(
                        {
                            "file_name": o_imgname,
                            "coco_url": "",
                            "height": o_PILimg.height,
                            "width": o_PILimg.width,
                            "date_captured": "",
                            "id": img_counter,
                        }
                    )
                    # Update masks
                    _, o_pred_mask_fullsize, n_inst = filter_and_save_instances(
                        imgname=o_imgname,
                        image=fullsize_overlap_image,
                        boxes=o_pred_boxes,
                        indexes=np.arange(len(o_labels)),
                        labels=o_labels,
                        masks=o_pred_mask_fullsize,
                        coco=coco,
                        data=data,
                        img_id=img_counter,
                        resolution=resolution,
                        crops_dir=CROPS_DIR,
                        minarea=minarea,
                        minsize=minsize,
                        rand_bg=False,
                        del_bg=False,
                        bgcolor=bgcolor,
                        save_crops=True,
                    )

                    print(
                        "...OK. Found {} instances, [{} objects touching the edges] ".format(
                            n_inst, n_inst - n_inst_before
                        )
                    )

                    # Save labels
                    o_labelname = "OVERLAP_{}-{}.png".format(
                        os.path.splitext(prev_imgname)[0], os.path.splitext(imgname)[0]
                    )
                    Image.fromarray(o_pred_mask_fullsize.astype(np.uint16)).save(os.path.join(LABELS_DIR, o_labelname))

                    if SAVE_IMGS:
                        vizuname = "VIZU-{}.jpg".format(os.path.splitext(o_imgname)[0])
                        o_resized_masks = sk.transform.resize(
                            o_pred_mask_fullsize,
                            overlap_image.shape[:-1],
                            order=0,
                            anti_aliasing=False,
                        )
                        vizu = label2rgb(o_resized_masks, overlap_image, alpha=0.25)
                        bd = sk.segmentation.find_boundaries(
                            o_resized_masks, connectivity=2, mode="inner", background=0
                        )
                        vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                        vizu = np.around(255 * vizu).astype(np.uint8)
                        Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        # paramètres utilisés dans l'itération suivante
        bottom_indexes = np.where(pred_boxes[:, 2] >= fullsize_nx - deltaL)
        # print("bottom boxes", pred_boxes[bottom_indexes])
        # bottom_labels = labels[bottom_indexes]
        prev_fullsize_bottom_boxes = pred_boxes[bottom_indexes]
        prev_fullsize_cleaned_image = fullsize_cleaned_image
        prev_imgname = imgname

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

def predict_onebyone(
    coco,
    imsize,
    input_dir,
    output_dir,
    model_file,
    resolution,
    thresholds=(0.5, 0.5, 0.6),
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=MINAREA,
    minsize=MINSIZE,
):
    """Instance segmentation on an image or a series of images.
    A SOLOv2 network is first used to predict the instance masks, downsampling input images if needed.

    Inputs:
    coco: a dict containing info, licence and categories in coco format
    imsize: size of input images
    input_dir: all the images in input_dir will be processed
    output_dir: where to put teh results
    model_file: path to tensorflow saved model
    resolution: res of the input images
    trhesolds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -COCO object (dict)
    -data dict

    Creates on disk:
    - A coco file (json)
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images  in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - individual instances in output_dir/crops folder
    - newly created overlap bands images (full resolution) in output_dir/images

    """
    # TODO: maybe compute regionprops on low res images and scale up results to speed things up ?

    # Load model config and weights
    tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    model_directory = os.path.dirname(model_file)
    with open(os.path.join(model_directory, "config.json"), "r", encoding="utf-8") as configfile:
        config = json.load(configfile)

    # Creating architecture using config file
    modelconf = SOLOv2.Config(**config)

    detector = SOLOv2.model.SOLOv2Model(modelconf)
    print(f"Loading model {model_file}...", end="")
    # Loading weights
    detector.load_weights(model_file, by_name=False, skip_mismatch=False).expect_partial()
    print("OK")

    # Retrieve images
    img_dict = {}
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

    coco["info"]["description"] = input_dir
    coco["annotations"] = []
    coco["images"] = []

    OUTPUT_DIR = os.path.join(output_dir, str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    CROPS_DIR = OUTPUT_DIR / Path("crops")
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")
    OVERLAPPING_IMGS_DIR = OUTPUT_DIR / Path("images")

    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(OVERLAPPING_IMGS_DIR, exist_ok=True)

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
    }

    nx, ny = imsize

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    for counter, imgname in enumerate(keys):
        print(
            "Processing image {} size ({}/{})".format(imgname, counter, len(keys)),
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
        # Resize if the ratio nx/ny is not the same (if not, bboxes are not rescaled correctly)
        fullsize_nx, fullsize_ny, _ = fullsize_image.shape
        if nx * fullsize_ny != fullsize_nx * ny:
            ratio = min(nx / fullsize_nx, ny / fullsize_ny)
            fullsize_image = tf.image.resize_with_pad(fullsize_image, int(nx / ratio), int(ny / ratio), antialias=True)
            fullsize_nx, fullsize_ny, _ = fullsize_image.shape

        # Get network predictions
        seg_preds, scores, cls_labels = detector(image[tf.newaxis, ...], training=False, **default_kwargs)

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

        pred_mask_fullsize = sk.transform.resize(
            labeled_masks, (fullsize_nx, fullsize_ny), order=0, anti_aliasing=False
        )

        # Extract bboxes
        region_properties = regionprops(pred_mask_fullsize)
        pred_boxes = np.array([prop["bbox"] for prop in region_properties])
        labels = np.array([prop["label"] for prop in region_properties])
        indexes = np.arange(labels.shape[0])

        if labels.shape[0] > 0:
            (
                fullsize_cleaned_image,
                pred_mask_fullsize,
                n_inst,
            ) = filter_and_save_instances(
                imgname=imgname,
                image=fullsize_image,
                boxes=pred_boxes,
                indexes=indexes,
                labels=labels,
                masks=pred_mask_fullsize,
                coco=coco,
                data=data,
                img_id=img_counter,
                resolution=resolution,
                crops_dir=CROPS_DIR,
                minarea=minarea,
                minsize=minsize,
                rand_bg=False,
                del_bg=False,
                bgcolor=bgcolor,
                save_crops=True,
            )

            print(f"...OK. Found {n_inst} instances. ")

            # Save labels
            labelname = "{}.png".format(os.path.splitext(imgname)[0])
            Image.fromarray(pred_mask_fullsize.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

            if SAVE_IMGS:
                vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
                resized_masks = sk.transform.resize(pred_mask_fullsize, (nx, ny), order=0, anti_aliasing=False)
                bd = sk.segmentation.find_boundaries(resized_masks, connectivity=2, mode="inner", background=0)
                vizu = label2rgb(resized_masks, image, alpha=0.25, colors=SOLOv2.visualization._COLORS)
                vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                vizu = np.around(255 * vizu).astype(np.uint8)
                Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))
        else:
            print("...OK. No instance found !")

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
