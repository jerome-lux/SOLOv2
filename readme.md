# SOLOv2

Tensorflow implementation of [SOLOv2](https://arxiv.org/pdf/2003.10152.pdf) (Segmenting Objects by LOcations) in full graph mode for better performance<br>
This implementation is partly inspired by https://www.fastestimator.org/

## Creating the model
First, create a config object

    config = SOLOv2.Config() #default config

You can also customize the config:

    params = {
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

When using a custom backbone, you have to put the name of the layers that will be connected to the FPN in the dict "connection_layers"

The model architecture can be accessed using the .model attribute

## Training with custom dataset: <br>
By default, the dataset is loaded using a custom DataLoader class<br>
The dataset files should be stored in 3 folders:<br>
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

The dataset attribute is a tf.dataset and it will output:
- image name,
- image [H W, 3],
- masks [H,W]: integer labeled instance image
- box [N]: box in xyxy formt for each instance
- cls_ids [N]: class id of each instance
- labels [N]: label (in the mask image) of each instance

The dataset is batched using tf.data.experimental.dense_to_ragged_batch.

It should be easy to create a DataLoader for other formats like COCO.

To train the model, use the "train" function, with the chosen optimizer, batch size and callbacks: <br>

    SOLOv2.model.train(mySOLOv2model,
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
A call to the model with a [1, H, W, 3] image returns the N masks tensor (one slice per instance [1, N, H/2, W/2]) and corresponding classes [1, N] and scores [1, N]. <br>
The model ALWAYS return ragged tensors, and should work with batchsize > 1.
The final labeled prediction can be obtained by the SOLOv2.utils.decode_predictions function

    seg_peds, cls_ids, scores = mySOLOv2model(input)
    labeled_masks = SOLOv2.utils.decode_predictions function(seg_preds, scores, threshold=0.5, by_scores=True)

Results can be vizualised using the SOLOv2.visualization.draw_instances function:

    img = SOLOv2.visualization.draw_instances(input, 
               labeled_masks.numpy(), 
               cls_ids=cls_labels[0,...].numpy() + 1, 
               cls_scores=scores[0,...].numpy(), 
               class_ids_to_name=id_to_cls, 
               show=True, 
               fontscale=0., 
               fontcolor=(0,0,0),
               alpha=0.5, 
               thickness=0)

Note that all inputs to this function must bhave a batch dimension an should be converted to numpy arrays.
    


