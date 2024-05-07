import torchvision 
import torch.nn as nn


"""
No weight sharing between the main model and the auxiliary model
"""


class PreModel(nn.Module):
    def __init__(self):
        super(PreModel, self).__init__()
        resnet_base_model = torchvision.models.resnet18(weights=None)

        self.model = nn.Sequential(*list(resnet_base_model.children())[:5])

    def forward(self, x):
        x = self.model(x)
        return x
    


class AuxModel(nn.Module):
    def __init__(self):
        super(AuxModel, self).__init__()
        resnet_base_model = torchvision.models.resnet18(weights=None)
        
        self.layer_1 = nn.Sequential(*list(resnet_base_model.children())[5])    #! Take only the second convolutional layer
        self.layer_2 = nn.AdaptiveAvgPool2d((1, 1))    #! Take the average of the output of the second convolutional layer
        self.layer_3 = nn.Flatten()    #! Flatten the output of the average pooling layer
        
        self.layer_4 = nn.Linear(128, 4) #! Output layer for the auxiliary model with 4 classes
        self.layer_5 = nn.Softmax()

        self.layer_6 = nn.Linear(128, 1) #! Output layer for the auxiliary model with 1 target
        self.layer_7 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        label = self.layer_4(x)
        label = self.layer_5(label)

        target = self.layer_6(x)
        target = self.layer_7(target)
        return label, target
    
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        resnet_base_model = torchvision.models.resnet18(weights=None)

        self.layer_1 = nn.Sequential(*list(resnet_base_model.children())[5]) #! Take the second convolutional layer
        self.layer_2 = nn.Sequential(*list(resnet_base_model.children())[6]) #! Take the third convolutional layer
        self.layer_3 = nn.Sequential(*list(resnet_base_model.children())[7]) #! Take the fourth convolutional layer
        self.layer_4 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_5 = nn.Flatten()

        self.layer_6 = nn.Linear(512, 4) #! Output layer for the main model with 256 neurons
    
        self.layer_7 = nn.Softmax()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        return x