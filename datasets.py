import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
#from natsort import natsorted
from torchvision.datasets import ImageFolder


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image






transform = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
   # T.RandomCrop(c.cropsize),
    T.Resize([c.imageSize, c.imageSize]), 
    T.ToTensor()
])

transform_val = T.Compose([
    #T.CenterCrop(c.cropsize_val),
    T.Resize([c.imageSize, c.imageSize]), 
    T.ToTensor(),
])

train_dataset = ImageFolder(
            c.TRAIN_PATH,
            transform)
val_dataset = ImageFolder(
            c.VAL_PATH,
            transform_val)

# Training data loader
trainloader = DataLoader(
    #Hinet_Dataset(transforms_=transform, mode="train"),
    train_dataset,
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    #Hinet_Dataset(transforms_=transform_val, mode="val"),
    val_dataset,
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)
transforms_color = T.Compose([
                T.Resize([128, 128]),
                T.ToTensor(),
            ])
transforms_cover = transforms_color
transforms_secret = transforms_color
test_v_dataset_cover = ImageFolder(
            c.coverdir,
            transforms_cover)
test_v_dataset_secret = ImageFolder(
            c.secretdir,
            transforms_secret)
test_v_loader_secret = DataLoader(test_v_dataset_secret, batch_size=c.batch_size*2,
                                        shuffle=True, num_workers=1)
test_v_loader_cover = DataLoader(test_v_dataset_cover, batch_size=c.batchsize_val,
                                       shuffle=True, num_workers=1)
test_v_loader = zip(test_v_loader_secret, test_v_loader_cover)