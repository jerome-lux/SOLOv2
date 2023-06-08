import json
from pathlib import Path
import os


class Config():

    def __init__(self, **kwargs):

        self.load_backbone = False
        self.backbone_params = {}
        self.backbone = 'resnext50',

        # Specific params
        self.ncls = 1
        self.imshape = (768, 1536, 3)

        # General layers params
        self.activation = 'gelu'
        self.normalization = "gn"  # gn for Group Norm
        self.normalization_kw = {'groups': 32}
        self.model_name = "SOLOv2-Resnext50"

        # FPN
        self.connection_layers = {"C2": "stage1_block3Convblock",
                                  "C3": "stage2_block4Convblock",
                                  "C4": "stage3_block6Convblock",
                                  "C5": "stage4_block3Convblock"}
        self.FPN_filters = 256
        self.extra_FPN_layers = 1  # layers after P5. Strides must correspond to the number of FPN layers !

        # SOLO head
        self.head_filters = [256, 256, 256, 256]  # Filters per stage
        self.strides = [4, 8, 16, 32, 64]
        self.head_layers = 4  # Number of repeats of head conv layers
        self.head_filters = 256
        self.kernel_size = 1
        self.grid_sizes = [64, 36, 24, 16, 12]
        self.scale_ranges = [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]]
        self.offset_factor = 0.25

        # SOLO MASK head
        self.mask_mid_filters = 128
        self.mask_output_filters = 256

        # loss
        self.lossweights = [1., 1.]

        # Update defaults parameters with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        s = ""

        for k, v in self.__dict__.items():
            s += "{}:{}\n".format(k, v)

        return s

    def save(self, filename):

        # data = {k:v for k, v in self.__dict__.items()}

        p = Path(filename).parent.absolute()
        if not os.path.isdir(p):
            os.mkdir(p)

        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)
