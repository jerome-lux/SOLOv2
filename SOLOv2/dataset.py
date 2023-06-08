import os
from PIL import Image
import json
import numpy as np
import tensorflow as tf
from . import utils

VALID_IMAGE_FORMATS = ['.jpg', '.png', '.tif', '.bmp']

# TODO:use ragged tensor for bboxes, cls, labels ?


VALID_IMAGE_FORMATS = ['.jpg', '.png', '.tif', '.bmp']


class DataLoader():

    """
    DataLoader class containing list of images, labeled masks, bboxes and classes and tf.data.Dataset
    folder: base folder containing images, masks and annotations in images/ labels/ and metadata/ subfolders
    cls_ind: dictionnary mapping classes names and indices
    mode: either 'tf' or 'py: if tf create a tf.Dataset generator else create a pure python generator
    """

    def __init__(self, folder, cls_ind):

        self.data_dir = os.path.join(folder, "annotations")
        self.images_dir = os.path.join(folder, "images")
        self.masks_dir = os.path.join(folder, "labels")

        # images ids and ids to img name dict
        self.imgs_ids = {os.path.splitext(f)[0]: i for i, f in enumerate(os.listdir(self.data_dir))
                         if os.path.isfile(os.path.join(self.data_dir, f))
                         and os.path.splitext(f)[-1].lower() == ".json" and not f.startswith('.')}

        self.ids_to_imgs = {v: k for k, v in self.imgs_ids.items()}

        self.data_dict = {k: os.path.join(
            self.data_dir, k + ".json") for k in self.imgs_ids}

        # List of imgs and masks (because we do not know the extensions)
        self.image_dict = {
            os.path.splitext(f)[0]: os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, f)) and os.path.splitext(f)[-1].lower() in
            VALID_IMAGE_FORMATS}

        self.mask_dict = {os.path.splitext(f)[0]: os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)
                          if os.path.isfile(os.path.join(self.masks_dir, f))
                          and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS}

        # reorder keys
        self.image_dict = {k: self.image_dict[k]
                           for k in self.data_dict.keys()}
        self.mask_dict = {k: self.mask_dict[k]
                          for k in self.data_dict.keys()}
        # dict: {'CLASSNAME':idx,...}
        self.cls_ind = cls_ind
        self.ncls = np.unique(list(self.cls_ind.values())).size
        self.length = len(self.image_dict)

        self.build()

    def build(self):
        """Build the tf.data.Dataset
        each element contains:
        basename, image, mask, bboxes, classes, labels
        """
        self.dataset = tf.data.Dataset.from_tensor_slices((list(self.image_dict.values()),
                                                           list(
            self.mask_dict.values()),
            list(self.data_dict.values())))
        self.dataset = self.dataset.map(
            lambda x, y, z: tf.py_function(
                func=self.parse_dataset, inp=[x, y, z],
                Tout=[
                    tf.TensorSpec(shape=(None,), dtype=tf.string),
                    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
                    tf.RaggedTensorSpec(shape=(None, None,), dtype=tf.int32),
                    tf.RaggedTensorSpec(shape=(None, None,), dtype=tf.int32),
                ]),
            num_parallel_calls=tf.data.AUTOTUNE)
        # Squeeze the unecessary dimension for boxes, class ids and box labels
        self.dataset = self.dataset.map(lambda a, b, c, d, e, f: (
            a, b, c, tf.squeeze(d, 0), tf.squeeze(e, 0), tf.squeeze(f, 0)),
            num_parallel_calls=tf.data.AUTOTUNE)

    def parse_dataset(self, imgfn, maskfn, datafn):
        """For an unkown reason, it is not possible to batch tensors with different length using
        tf.data.experimental.dense_to_ragged_batch
        It is necessary to return RaggedTensor for boxes, class indices and labels, but
        to do so, we must add a leading dimension
        if not, tensorflow throw an error "The rank of a RaggedTensor must be greater than 1"
        the extra dim must be deleted later...
        """

        # Image is already normalized when using tf.io
        image = tf.io.decode_image(tf.io.read_file(imgfn), channels=3, dtype=tf.float32)

        mask = np.array(Image.open(maskfn.numpy())).astype(np.int32)
        # mask = tf.io.decode_image(tf.io.read_file(maskfn), channels=1)
        nx, ny = tf.shape(image)[0], tf.shape(image)[1]
        mask = tf.image.resize(mask[...,tf.newaxis], size=(nx//2, ny//2), method='nearest')
        # print(nx,ny)

        with open(datafn.numpy(), "r") as jsonfile:
            data = json.load(jsonfile)
            bboxes = np.array([v['bbox'] for v in data.values()]).astype(np.float32)
            try:
                bboxes = utils.normalize_bboxes(bboxes, nx, ny)
            except:
                tf.print("Error, cannot read boxes", bboxes, imgfn.numpy())
            classes = np.array([self.cls_ind[v['class']]
                                for v in data.values()]).astype(np.int32)
            labels = np.array(list(data.keys())).astype(np.int32)

        return os.path.splitext(os.path.basename(imgfn.numpy()))[0], \
            image, \
            mask[..., 0], \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(bboxes, dtype=tf.float32)[tf.newaxis, ...]), \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(classes, dtype=tf.int32)[tf.newaxis, ...]), \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(labels, dtype=tf.int32)[tf.newaxis, ...])

    def get_img(self, imgid, normalize=True):
        key = self.ids_to_imgs[imgid]
        if normalize:
            return np.array(Image.open(self.image_dict[key])) / 255.

        return np.array(Image.open(self.image_dict[key]))

    def get_mask(self, imgid):
        key = self.ids_to_imgs[imgid]
        return np.array(Image.open(self.mask_dict[key]))

    def get_data(self, imgid):
        """Return instances data of image imgid, ie. a dict keyed by instance label and giving bboxes and class
        """
        key = self.ids_to_imgs[imgid]
        with open(self.data_dict[key], "r") as jsonfile:
            data = json.load(jsonfile)

        classes = np.array([self.cls_ind[v['class']]
                            for v in data.values()]).astype(np.int32)
        labels = np.array(list(data.keys())).astype(np.int32)
        bboxes = np.array([v['bbox'] for v in data.values()])

        return bboxes, classes, labels

    def get_bboxes(self, imgid):

        data = self.get_data(imgid)
        bboxes = np.array([v['bbox'] for v in data.values()])
        return bboxes