import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import AutoFeatureExtractor, TimesformerModel, AutoImageProcessor

class ViTB16MyModel(nn.Module) :
    def __init__(
        self,
        num_classes,
        loss={"xent", "htri"},
        pretrained=True,         
        **kwargs,
    ):
        super().__init__() # remember always write this line --- this is for pytorch ma
        # note, this is new implementation of how to load pre-trained weights
        
        model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        print(model)
        # assign this loaded model to self.model
        self.model = model
        # print (model) # you can always print the model, check out the classifier/head la
        feature_dim = 768
        # from the print result i find that the feature dim of ViT Base
        self.num_classes = num_classes
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(feature_dim, num_classes) #
        # model = vit b_ 16(pretrained-True)

    def forward (self, x):
        ##################################
        # Reshape and permute the input tensor
        x = self.model(x)
        x = x[0][:,0] 
        y = self.classifier(x)
        #v = x[:, 0]
        ################################
        return y
        

def vitb16(num_classes, pretrained=True):
    model = ViTB16MyModel(
        # "vitB16",
        num_classes=num_classes,
        pretrained=pretrained,        
    )
    return model