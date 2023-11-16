import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
from random import randrange

class HRImageDataset(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        self.img_height = img_height
        self.img_width  = img_width
        
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.resize_full = transforms.Resize((img_height, img_width), Image.BICUBIC)
        self.folders = sorted(glob.glob(root + "/*"))
    def __getitem__(self, index):
        index = index % len(self.folders)
        img    = Image.open(self.folders[index] + "/Image.png").convert(mode="RGB")
        depth  = Image.open(self.folders[index] + "/Depth.png").convert(mode="RGB")
        normal = Image.open(self.folders[index] + "/Normal.png").convert(mode="RGB")
        obj    = Image.open(self.folders[index] + "/Object.png").convert(mode="RGB")
        
        img    = self.tensor_transform(img)
        depth  = self.tensor_transform(depth)
        normal = self.tensor_transform(normal)
        obj    = self.tensor_transform(obj)
        
        result = torch.cat((img, depth[0:1], normal, obj[0:1]))
        
        #image 0-2
        #depth 3
        #normal 4-6
        #object 7
        hr = self.resize_full(result)
        
        return hr
    
    def __len__(self):
        return len(self.folders)


class ImageDataset(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        self.img_height = img_height
        self.img_width  = img_width
        
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.randoms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        )
        self.crop_small = transforms.RandomCrop(size=(img_height, img_width))
        self.crop_mid   = transforms.RandomCrop(size=(img_height + img_height // 2, img_width + img_width // 2))
        self.crop_large = transforms.RandomCrop(size=(img_height * 2, img_width * 2))
        self.resize_full = transforms.Resize((img_height, img_width), Image.BICUBIC)
        self.resize_small = transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC)
        self.folders = sorted(glob.glob(root + "/*"))
    def __getitem__(self, index):
        index = index % len(self.folders)
        img    = Image.open(self.folders[index] + "/Image.png").convert(mode="RGB")
        depth  = Image.open(self.folders[index] + "/Depth.png").convert(mode="RGB")
        normal = Image.open(self.folders[index] + "/Normal.png").convert(mode="RGB")
        obj    = Image.open(self.folders[index] + "/Object.png").convert(mode="RGB")
        
        img    = self.tensor_transform(img)
        depth  = self.tensor_transform(depth)
        normal = self.tensor_transform(normal)
        obj    = self.tensor_transform(obj)
        
        result = torch.cat((img, depth[0:1], normal, obj[0:1]))
        result = self.randoms(result)
        r = randrange(0, 3)
        if r == 0:
            result = self.crop_small(result)
        elif r == 1:
            result = self.crop_mid(result)
        else:
            result = self.crop_large(result)
        
        #image 0-2
        #depth 3
        #normal 4-6
        #object 7
        hr = self.resize_full(result)
        lr = self.resize_small(result)
        
        return {"lr":lr, "hr":hr}
    
    def __len__(self):
        return len(self.folders)
    
class SRGANDataset(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        self.img_height = img_height
        self.img_width  = img_width
        
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.randoms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        )
        self.crop_small = transforms.RandomCrop(size=(img_height, img_width))
        self.crop_mid   = transforms.RandomCrop(size=(img_height + img_height // 2, img_width + img_width // 2))
        self.crop_large = transforms.RandomCrop(size=(img_height * 2, img_width * 2))
        self.resize_full = transforms.Resize((img_height, img_width), Image.BICUBIC)
        self.resize_small = transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC)
        self.folders = sorted(glob.glob(root + "/*"))
    def __getitem__(self, index):
        index = index % len(self.folders)

        r = randrange(0, 4)
        img = None
        if r == 0:
            img = Image.open(self.folders[index] + "/Image.png").convert(mode="RGB")
        elif r == 1:
            img = Image.open(self.folders[index] + "/Depth.png").convert(mode="RGB")
        elif r == 2:
            img = Image.open(self.folders[index] + "/Normal.png").convert(mode="RGB")
        else:
            img = Image.open(self.folders[index] + "/Object.png").convert(mode="RGB")
        
        img    = self.tensor_transform(img)
        result = self.randoms(img)

        r = randrange(0, 3)
        if r == 0:
            result = self.crop_small(result)
        elif r == 1:
            result = self.crop_mid(result)
        else:
            result = self.crop_large(result)
        
        hr = self.resize_full(result)
        lr = self.resize_small(result)
        
        return {"lr":lr, "hr":hr}
    
    def __len__(self):
        return len(self.folders)

class AllMetaDataset(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        self.img_height = img_height
        self.img_width  = img_width
        
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.randoms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        )
        self.crop_small = transforms.RandomCrop(size=(img_height, img_width))
        self.crop_mid   = transforms.RandomCrop(size=(img_height + img_height // 2, img_width + img_width // 2))
        self.crop_large = transforms.RandomCrop(size=(img_height * 2, img_width * 2))
        self.resize_full = transforms.Resize((img_height, img_width), Image.BICUBIC)
        self.resize_small = transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC)
        self.folders = sorted(glob.glob(root + "/*"))
    def __getitem__(self, index):
        index = index % len(self.folders)
        
        img    = Image.open(self.folders[index] + "/Image.png").convert(mode="RGB")
        depth  = Image.open(self.folders[index] + "/Depth.png").convert(mode="RGB")
        normal = Image.open(self.folders[index] + "/Normal.png").convert(mode="RGB")
        obj    = Image.open(self.folders[index] + "/Object.png").convert(mode="RGB")
        
        img    = self.tensor_transform(img)
        depth  = self.tensor_transform(depth)
        normal = self.tensor_transform(normal)
        obj    = self.tensor_transform(obj)

        result = torch.cat((img, depth, normal, obj), 0)
        result = self.randoms(result)

        hr = self.resize_full(result)
        lr = self.resize_small(result)
        
        return {
            "lr_img":lr[0:3], 
            "hr_img":hr[0:3], 
            "lr_depth":lr[3:6], 
            "hr_depth":hr[3:6],
            "lr_normal":lr[6:9],
            "hr_normal":hr[6:9],
            "lr_obj":lr[9:12],
            "hr_obj":hr[9:12]
        }
    
    def __len__(self):
        return len(self.folders)