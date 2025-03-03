import torch
import numpy as np
import pandas as pd
import nibabel as nib
import monai.transforms as T
from torch.utils.data import Dataset, DataLoader
# from imgaug import augmenters as iaa

def get_surv_array(time, event, intervals):
    """
    Transforms censored survival data into vector format that can be used in Keras.
    Args:
        time: Array of failure/censoring times.
        event: Array of censoring indicator. 1 if failed, 0 if censored.
        intervals: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Return:
        surv_array: Dimensions with (number of samples, number of time intervals*2)
    """
    
    breaks = np.array(intervals)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    
    surv_array = np.zeros((n_intervals * 2))
    
    if event == 1:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks[1:]) 
        if time < breaks[-1]:
            surv_array[n_intervals + np.where(time < breaks[1:])[0][0]] = 1
    else:
        surv_array[0 : n_intervals] = 1.0 * (time >= breaks_midpoint)
    
    return surv_array

def split_samples(samples_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """根据训练集：验证集：测试集的比例划分samples"""
    assert train_ratio + val_ratio + test_ratio == 1, "train_ratio + val_ratio + test_ratio should be 1"
    samples = pd.read_csv(samples_path, encoding='utf-8').iloc[:, 0].tolist()
    total_samples = len(samples)
    train_samples = samples[:int(total_samples * train_ratio)]
    val_samples = samples[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
    test_samples = samples[int(total_samples * (train_ratio + val_ratio)):]
    return train_samples, val_samples, test_samples

def get_transforms():
    """获取数据预处理的transforms"""
    return T.Compose([
            T.CenterSpatialCrop(roi_size=112),
            T.Resize((112, 112, 112)),
            T.ToTensor(),
        ])

"""Dataset for preprocessed images"""
class MyDataset(Dataset):
    def __init__(self, root, tabular, samples, intervals, mode='train', transform=None, seed=0):
        self.root = root
        self.tabular = tabular
        self.samples = samples
        self.intervals = intervals
        self.mode = mode
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt, ct = torch.zeros(0), torch.zeros(0)
        image_list = []

        pt = nib.load(f'{self.root}/pet/{self.samples[idx]}_pet.nii.gz').get_fdata()
        pt = pt[np.newaxis, ...]
        # pt preprocess
        # pt[pt < 100] = 0
        # pt = self.norm(pt)
        image_list.append(pt)
        
        ct = nib.load(f'{self.root}/ct/{self.samples[idx]}_ct.nii.gz').get_fdata()
        # ct = self.norm(ct)
        ct = ct[np.newaxis, ...]
        image_list.append(ct)
        
        # if self.transform:
        #     image_list = [self.transform(np.squeeze(image, axis=0)).float() for image in image_list]
        if len(image_list) == 3:
            pt, ct, seg = image_list
        else:
            pt, ct = image_list
            
        pt = pt.astype(np.float32)
        ct = ct.astype(np.float32)
        pt = torch.from_numpy(pt).float()
        ct = torch.from_numpy(ct).float()

        df = pd.read_csv(f'{self.root}/{self.tabular}')
        line = df[df.iloc[:, 0] == self.samples[idx]]
        tabular = torch.from_numpy(line.iloc[:, 1:-2].values).float().squeeze(0)
        time = line['PFS/M'].values
        event = line['censorship'].values
        surv_array = torch.from_numpy(get_surv_array(time, event, self.intervals)).float().squeeze(0)
        time = torch.from_numpy(time).float().squeeze(0)
        event = torch.from_numpy(event).float().squeeze(0)
        
        return pt, ct, tabular, time, event, surv_array, idx

    def norm(self, image, lower_percentile=0.5, upper_percentile=99.5):
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)
        image = np.clip(image, lower_bound, upper_bound)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
        return image
    
"""Dataset for original images"""
class MyDataset_origin(Dataset):
    def __init__(self, root, tabular, samples, intervals, mode='train', transform=None, seed=0):
        self.root = root
        self.tabular = tabular
        self.samples = samples
        self.intervals = intervals
        self.mode = mode
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt, ct = torch.zeros(0), torch.zeros(0)
        image_list = []
        
        # try:
        #     pt = nib.load(f'{self.root}/images_preprocessed/{self.samples[idx]}_PET.nii.gz').get_fdata()
        #     pt = pt[np.newaxis, np.newaxis, ...]
        #     #pt preprocess
        #     pt[pt<100] = 0
        #     pt = self.norm(pt)
        #     image_list.append(pt)
        # except FileNotFoundError as e:
        #     pass
        pt = nib.load(f'{self.root}/pet/{self.samples[idx]}_pet.nii.gz').get_fdata()
        pt = pt[np.newaxis, np.newaxis, ...]
        # pt preprocess
        pt[pt < 100] = 0
        pt = self.norm(pt)
        image_list.append(pt)
        
        ct = nib.load(f'{self.root}/ct/{self.samples[idx]}_ct.nii.gz').get_fdata()
        ct = self.norm(ct)
        ct = ct[np.newaxis, np.newaxis, ...]
        image_list.append(ct)
        
        # seg = nib.load(f'{self.root}/images_preprocessed/{self.samples[idx]}__Seg.nii.gz').get_fdata()
        # seg = seg[np.newaxis, np.newaxis, ...]
        # seg = np.where(seg == 1, 1, 0)
        # image_list.append(seg)
        
        # if self.mode == 'train':
        #     image_list = self.augmentation(image_list)
        if self.transform:
            image_list = [self.transform(np.squeeze(image, axis=0)).float() for image in image_list]
        if len(image_list) == 3:
            pt, ct, seg = image_list
        else:
            pt, ct = image_list
        
        df = pd.read_csv(f'{self.root}/{self.tabular}')
        line = df[df.iloc[:, 0] == self.samples[idx]]
        tabular = torch.from_numpy(line.iloc[:, 1:-2].values).float().squeeze(0)
        time = line['PFS/M'].values
        event = line['censorship'].values
        surv_array = torch.from_numpy(get_surv_array(time, event, self.intervals)).float().squeeze(0)
        time = torch.from_numpy(time).float().squeeze(0)
        event = torch.from_numpy(event).float().squeeze(0)
        
        return pt, ct, tabular, time, event, surv_array, idx

    def norm(self, image, lower_percentile=0.5, upper_percentile=99.5):
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)
        image = np.clip(image, lower_bound, upper_bound)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
        return image

    
    # def augmentation(self, image_list):
    #     aug_seq = iaa.Sequential([
    #         iaa.Affine(translate_percent={"x": [-0.1, 0.1], "y": [0, 0]},
    #                    scale={"x": (0.9, 1.1), "y": (1.0, 1.0)},
    #                    shear=(-10, 10),
    #                    rotate=(-10, 10)),
    #         iaa.CenterCropToFixedSize(width=112, height=None)
    #         ], random_order=False)
        
    #     n = len(image_list)
        
    #     """pre-process data shape"""
    #     for i in range(n):
    #         image_list[i] = image_list[i][:, 0, :, :, :]
        
    #     """flip/translate in x axls, rotate along z axls"""
    #     images = np.concatenate(image_list, axis=-1)
    #     images_aug = np.array(aug_seq(images=images))
        
    #     for i in range(n):
    #         image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
    #         image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
    #     """translate in z axls, rotate along y axls"""
    #     images = np.concatenate(image_list, axis=-1)
    #     images_aug = np.array(aug_seq(images=images))
        
    #     for i in range(n):
    #         image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
    #         image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
    #     """translate in y axls, rotate along x axls"""
    #     images = np.concatenate(image_list, axis=-1)
    #     images_aug = np.array(aug_seq(images=images))
        
    #     """recover axls"""
    #     for i in range(n):
    #         image_list[i] = images_aug[..., int(images_aug.shape[3]/n)*i:int(images_aug.shape[3]/n)*(i+1)]
    #         image_list[i] = np.transpose(image_list[i], (0, 3, 1, 2))
        
    #     # """reset Seg mask to 1/0"""
    #     # for i in range(image_list[-1].shape[0]):
    #     #     _, image_list[-1][i] = cv2.threshold(image_list[-1][i], 0.2, 1, cv2.THRESH_BINARY)
        
    #     """post-process data shape"""
    #     for i in range(n):
    #         image_list[i] = image_list[i][..., np.newaxis].transpose((0, 4, 1, 2, 3))
        
    #     return image_list