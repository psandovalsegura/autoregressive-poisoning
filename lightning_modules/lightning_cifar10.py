import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from pytorch_lightning.core.lightning import LightningModule

from models import *
from augmentations import MixUp, CutMix, Cutout, CutMixCrossEntropyLoss
from poison_datasets import AdversarialPoison, UnlearnablePoison

class LitCIFAR10Model(LightningModule):
    def __init__(self, 
                 model_name, 
                 batch_size=128, 
                 num_workers=16, 
                 learning_rate=0.025,
                 weight_decay=5e-4,
                 momentum=0.9,
                 adversarial_poison_path=False,
                 unlearnable_poison_path=False,
                 base_dataset_path=None,
                 augmentations_key=None):
        super().__init__()
        self.model = get_model_class_from_name(model_name=model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adversarial_poison_path = adversarial_poison_path
        self.unlearnable_poison_path = unlearnable_poison_path
        self.base_dataset_path = base_dataset_path
        self.augmentations_key = augmentations_key
        self.loss_fn = self.configure_criterion()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="test")

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / len(y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def configure_dataset(self, dataset):
        if 'mixup' in self.augmentations_key:
            dataset = MixUp(dataset, num_class=10)
        elif 'cutmix' in self.augmentations_key:
            dataset = CutMix(dataset, num_class=10)
        return dataset

    def configure_criterion(self):
        if self.augmentations_key in ['mixup', 'cutmix']:
            return CutMixCrossEntropyLoss()
        return nn.CrossEntropyLoss()

    def configure_transform(self, transform):
        if 'cutout' in self.augmentations_key:
            transform.transforms.append(Cutout(16))
        return transform

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.learning_rate, 
                              momentum=self.momentum, 
                              weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = self.configure_transform(transform_train)
        trainset = datasets.CIFAR10(
            root=self.base_dataset_path, train=True, download=False, transform=transform_train)
        if self.adversarial_poison_path:
            trainset = AdversarialPoison(root=self.adversarial_poison_path, baseset=trainset)
        if self.unlearnable_poison_path:
            trainset = UnlearnablePoison(root=self.unlearnable_poison_path, baseset=trainset)
        trainset = self.configure_dataset(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return trainloader

    def val_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(
            root=self.base_dataset_path, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return testloader

    def test_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(
            root=self.base_dataset_path, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return testloader

def get_model_class_from_name(model_name):
    if str.lower(model_name) == "vgg19":
        return VGG('VGG19')
    elif str.lower(model_name) == "resnet18":
        return ResNet18()
    elif str.lower(model_name) == "preactresnet18":
        return PreActResNet18()
    elif str.lower(model_name) == "googlenet":
        return GoogLeNet()
    elif str.lower(model_name) == "densenet121":
        return DenseNet121()    
    elif str.lower(model_name) == "resnext29_2x64d":
        return ResNeXt29_2x64d()
    elif str.lower(model_name) == "mobilenet":
        return MobileNet()
    elif str.lower(model_name) == "mobilenetv2":
        return MobileNetV2()
    elif str.lower(model_name) == "dpn92":
        return DPN92()
    elif str.lower(model_name) == "shufflenetg2":
        return ShuffleNetG2()
    elif str.lower(model_name) == "senet18":
        return SENet18()
    elif str.lower(model_name) == "efficientnetb0":
        return EfficientNetB0()
    elif str.lower(model_name) == "regnetx_200mf":
        return RegNetX_200MF()
    elif str.lower(model_name) == "convmixer":
        return ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=10)
    elif str.lower(model_name) == "vit_patch_2":
        return ViT(
            image_size = 32,
            patch_size = 2,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif str.lower(model_name) == "vit_patch_4":
        return ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
