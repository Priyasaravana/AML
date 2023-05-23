# Copyright (c) EEEM071, University of Surrey

from .ViTimesFormer import vitb16

__model_factory = {
    # image classification models
    "vitb16": vitb16,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    num_classes = 25
    pretrained = True
    return __model_factory[name](num_classes, pretrained)