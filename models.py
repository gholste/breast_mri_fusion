import torch
from torchvision import models
from albumentations.augmentations.transforms import Blur, HorizontalFlip, ElasticTransform, RandomScale, Resize, Rotate, RandomContrast
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch.transforms import ToTensor

class ShallowFFNN(torch.nn.Module):
    def __init__(self, meta_features):
        super(ShallowFFNN, self).__init__()
        self.meta_features = meta_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.meta_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, meta):
        return self.classifier.forward(meta)

class ResNet50FeatureExtractor(torch.nn.Module):
    def __init__(self, pre_trained=False, frozen=False):
        super(ResNet50FeatureExtractor, self).__init__()
        self.cnn = models.resnet50(pretrained=pre_trained)

        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.n_features_out = self.cnn.fc.in_features
        del(self.cnn.fc)

        if frozen:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNet50(torch.nn.Module):
    def __init__(self, pre_trained=True, frozen=False):
        super(ResNet50, self).__init__()
        self.cnn = ResNet50FeatureExtractor(pre_trained=pre_trained, frozen=frozen)
       
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.n_features_out, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im):
        out = self.cnn.forward(im)
        out = self.classifier.forward(out)

        return out

class FeatureFusion(torch.nn.Module):
    def __init__(self, meta_features, pre_trained=False, frozen=False):
        super(FeatureFusion, self).__init__()

        self.meta_features = meta_features
        self.cnn = ResNet50FeatureExtractor(pre_trained=pre_trained, frozen=frozen)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.n_features_out + self.meta_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im, meta):
        x1 = self.cnn.forward(im)
        x2 = meta

        x = torch.cat((x1, x2), dim=-1)
        x = self.classifier.forward(x)

        return x

class LearnedFeatureFusion(torch.nn.Module):
    def __init__(self, meta_features, mode="concat", pre_trained=True, frozen=False):
        super(LearnedFeatureFusion, self).__init__()
        assert mode in ["concat", "multiply", "add"], "mode must be one of ['concat', 'multiply', 'add']"

        self.meta_features = meta_features
        self.mode = mode

        self.cnn = ResNet50FeatureExtractor(pre_trained=pre_trained, frozen=frozen)
        self.cnn.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True)
        )
        self.meta_nn = torch.nn.Sequential(
            torch.nn.Linear(self.meta_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512*2 if self.mode == "concat" else 512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im, meta):
        x1 = self.cnn.forward(im)
        x1 = self.cnn.fc.forward(x1)
        x2 = self.meta_nn.forward(meta)

        if self.mode == "concat":
            x = torch.cat((x1, x2), dim=-1)
        if self.mode == "multiply":
            x = x1*x2
        if self.mode == "add":
            x = x1+x2
        x = self.classifier.forward(x)

        return x

class ProbabilityFusion(torch.nn.Module):
    def __init__(self, meta_features, mode="concat", pre_trained=True, frozen=False):
        super(ProbabilityFusion, self).__init__()
        assert mode in ["concat", "multiply", "add"], "mode must be one of ['concat', 'multiply', 'add']"

        self.meta_features = meta_features
        self.mode = mode
        self.cnn = ResNet50(pre_trained=pre_trained, frozen=frozen)
        self.meta_nn = ShallowFFNN(meta_features=self.meta_features)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(16, 1)
        )

    def forward(self, im, meta):
        x1 = torch.sigmoid(self.cnn.forward(im))
        x2 = torch.sigmoid(self.meta_nn.forward(meta))

        if self.mode == "concat":
            x = torch.cat((x1, x2), dim=-1)
        if self.mode == "multiply":
            x = x1*x2
        if self.mode == "add":
            x = x1+x2
        x = self.classifier.forward(x)

        return x

class LearnedFeatureFusionVariant(torch.nn.Module):
    def __init__(self, meta_features, mode="concat", pre_trained=True, frozen=False):
        super(LearnedFeatureFusionModel, self).__init__()
        assert mode in ["concat", "multiply", "add"], "mode must be one of ['concat', 'multiply', 'add']"

        self.meta_features = meta_features
        self.mode = mode

        self.cnn = ResNet50FeatureExtractor(pre_trained=pre_trained, frozen=frozen)
        self.cnn.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True)
        )

        self.cnn_cls = torch.nn.Linear(512, 1)

        self.meta_nn = torch.nn.Sequential(
            torch.nn.Linear(self.meta_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )

        self.meta_nn_cls = torch.nn.Linear(512, 1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512*2 if self.mode == "concat" else 512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )


    def forward(self, im, meta):
        f_i = self.cnn.forward(im)
        f_i = self.cnn.fc.forward(f_i)
        f_m = self.meta_nn.forward(meta)

        if self.mode == "concat":
            f_f = torch.cat((f_i, f_m), dim=-1)
        if self.mode == "multiply":
            f_f = f_i*f_m
        if self.mode == "add":
            f_f = f_i+f_m

        y_f = self.classifier(f_f)
        y_i = self.cnn_cls(f_i)
        y_m = self.meta_nn_cls(f_m)

        return {'img_out': y_i, 'meta_out': y_m, 'fusion_out': y_f}
