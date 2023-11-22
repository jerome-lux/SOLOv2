import argparse
import os
import SOLOv2
import datetime
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)
print(gpus)
print(tf.__version__)

SAVE_IMGS = True
VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
MINAREA = 2048
MINSIZE = 32
BGCOLOR = (94, 160, 220)
deltaL = 10

now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

DEFAULT_COCO = {
    "info": {
        "year": datetime.datetime.now().strftime("%Y"),
        "version": 1,
        "description": "none",
        "contributor": "J.LUX",
        "url": "",
        "date_created": now,
    },
    "licenses": {},
    "categories": [
        {"supercategory": "Recycled_Aggregates", "id": 1, "name": "Ra"},
        {"supercategory": "Recycled_Aggregates", "id": 2, "name": "Rb01"},
        {"supercategory": "Recycled_Aggregates", "id": 3, "name": "Rb02"},
        {"supercategory": "Recycled_Aggregates", "id": 4, "name": "Rc"},
        {"supercategory": "Recycled_Aggregates", "id": 5, "name": "Ru01"},
        {"supercategory": "Recycled_Aggregates", "id": 6, "name": "Ru02"},
        {"supercategory": "Recycled_Aggregates", "id": 7, "name": "Ru03"},
        {"supercategory": "Recycled_Aggregates", "id": 8, "name": "Ru04"},
        {"supercategory": "Recycled_Aggregates", "id": 9, "name": "Ru05"},
        {"supercategory": "Recycled_Aggregates", "id": 10, "name": "Ru06"},
        {"supercategory": "Recycled_Aggregates", "id": 11, "name": "X01"},
        {"supercategory": "Recycled_Aggregates", "id": 12, "name": "X02"},
        {"supercategory": "Recycled_Aggregates", "id": 13, "name": "X03"},
        {"supercategory": "Recycled_Aggregates", "id": 14, "name": "X04"},
        {"supercategory": "Recycled_Aggregates", "id": 15, "name": "X05"},
        {"supercategory": "Recycled_Aggregates", "id": 16, "name": "X06"},
        {"supercategory": "Recycled_Aggregates", "id": 17, "name": "Rg"},
        {"supercategory": "Recycled_Aggregates", "id": 18, "name": "Pl"},
        {"supercategory": "Recycled_Aggregates", "id": 19, "name": "UNKNOWN"},
    ],
    "images": [],
    "annotations": [],
}


parser = argparse.ArgumentParser(
    description="Segment objects using SOLOv2 model. \n"
    "- Create a COCO ann file \n"
    "- Save crops and masks of each detected objets\n"
    "- If an object is cut off at the edge of the image, it create a new image using two consecutive images to get only complete objects\n"
    "- Low res image with colored object can be generated for visualization purpose",
    fromfile_prefix_chars="@",
)

parser.add_argument(
    "-i",
    "--input",
    dest="input_dir",
    default=os.getcwd(),
    help="input directory where images are stored. Default: current working directory",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output_dir",
    default=os.path.join(os.getcwd(), "_predict_" + now),
    help="output directory. Default: current working directory",
)
parser.add_argument("-m", "--model", dest="model_file", help="full path of the model file")
parser.add_argument(
    "--size",
    dest="imsize",
    nargs=2,
    default=(768, 1536),
    type=int,
    help="Size of the output image in pixels: nx, ny. Input image are resized to this size before being processed by the model",
)
parser.add_argument("--bgcolor", dest="bgcolor", nargs=3, default=BGCOLOR, type=int, help="bgcolor")
parser.add_argument(
    "--minarea",
    dest="minarea",
    default=MINAREA,
    type=int,
    help="Instance Area threshold",
)
parser.add_argument(
    "--minsize",
    dest="minsize",
    default=MINSIZE,
    type=int,
    help="Instance's box minimum size",
)
parser.add_argument(
    "--thresholds",
    dest="thresholds",
    nargs=3,
    default=(0.5, 0.5, 0.6),
    type=float,
    help="Model score threshold, mask threshold and threshold used in matrix NMS (default: (0.5, 0.5, 0.6))",
)
parser.add_argument("--max_inst", dest="max_detections", default=400, type=int, help="max_detections")
parser.add_argument(
    "--res",
    dest="resolution",
    default=np.nan,
    type=float,
    help="resolution of the input image",
)
parser.add_argument(
    "--delta",
    dest="deltaL",
    default=10,
    type=int,
    help="An object with a distance to edge less than this value is considered to be touching the edge. Default 10 pixels",
)

parser.add_argument(
    "--weight_by_scores",
    dest="weight_by_scores",
    action="store_true",
    help="weight masks by scores (when two masks overlap at a given position, keeps only the mask with higher score)",
)

args = parser.parse_args()

if __name__ == "__main__":
    SOLOv2.predict(coco=DEFAULT_COCO, **vars(args))
