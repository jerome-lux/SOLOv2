# SOLOv2

Tensorflow implementation of [SOLOv2] https://arxiv.org/pdf/2003.10152.pdf in full graph mode for better performance<br>

## Creating the model
First, create a config object

    config = SOLOv2.Config() #default config

You can also customize the config:

    params = {}
    "load_backbone":False,
    "backbone_params":{},
    "backbone":'resnext50',
    "ncls":1,
    "imshape":(768, 1536, 3),
    "activation:"gelu",
    "normalization":"gn",
    "normalization_kw":{'groups': 32},
    "model_name":"SOLOv2-Resnext50",
    "connection_layers":{'C2': 'stage1_block3Convblock', 'C3': 'stage2_block4Convblock', 'C4': 'stage3_block6Convblock', 'C5': 'stage4_block3Convblock'},
    "FPN_filters":256,
    "extra_FPN_layers":1,
    "head_filters":256,
    "strides":[4, 8, 16, 32, 64],
    "head_layers":4,
    "kernel_size":1,
    "grid_sizes":[64, 36, 24, 16, 12],
    "scale_ranges":[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
    "offset_factor":0.25,
    "mask_mid_filters":128,
    "mask_output_filters":256,
    "lossweights":[1.0, 1.0],
    }

     config = SOLOv2.Config(**params)

The backbone can be loaded using load_backbone=True and backbone="path_to_your_backbone". It is a resnext50 by default.<br>
Then create the model:

    mySOLOv2model = SOLOv2.model.SOLOv2Model(config)

## Training with custom dataset: <br>
Files are stored in 3 folders:
/images: RGB images <br>
/labels: labeled masks grey-level images (8, 16 or 32 bits int / uint) in **non compressed** format (png or bmp) <br>
/annotations: one json file per image containing a dict of dicts keyed by labels, with the class, and box coordinates in [x0, y0, x1, y1] format <br>

    '{"1": {"class": "cat", "bbox": [347, 806, 437, 886]}, "2": {"class": "dog", "bbox": [331, 539, 423, 618]}, ...}'

Note that each corresponding image, label and annotation file must have the same base name<br>

First create a dict with class names and index <br>

    cls_ind = {
    "background":0,
    "cat":1,
    "dog":2,
    ...
    }

then create a dataloader <br>

    trainset = SOLOv2.DataLoader("DATASET_PATH",cls_ind=cls_ind)

use the train method, with chosen optimizer, batch size and callbacks <br>

    SOLOv2.model.train(detector,
                       trainset.dataset,
                       epochs=epochs,
                       val_dataset=None,
                       steps_per_epoch=len(trainset.dataset) // batch_size,
                       validation_steps= 0,
                       batch_size=batch_size,
                       callbacks = callbacks,
                       optimizer=optimizer,
                       prefetch=tf.data.AUTOTUNE,
                       buffer=150)

## Inference
A call to the model with a [1,H,W,3] image returns the N one-hot masks tensor (one slice per instance [1, N, H/2, W/2]) and corresponding classes [1, N] and scores [1, N]. <br>
The inference works with batch size > 1 and the model then returns ragged tensors

The model architecture can be accessed using the .model attribute


