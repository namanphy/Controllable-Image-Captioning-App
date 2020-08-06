import os
import h5py
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(cfg):
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
    train_dataset = CaptionDataset(cfg.data_folder, cfg.data_name, 'TRAIN', 
                              transform = transforms.Compose([normalize]))
    val_dataset = CaptionDataset(cfg.data_folder, cfg.data_name, 'VAL', 
                                 transform = transforms.Compose([normalize]))
    test_dataset = None

    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, 
                              shuffle = True, num_workers = cfg.workers, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = cfg.batch_size, 
                            shuffle = False, num_workers = cfg.workers, pin_memory = True)
    test_loader = None
    return train_loader, val_loader, test_loader


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder='', data_name='', split='', img_file='', transform=None, inference=False):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.inference = inference
        if self.inference:
            # img = cv2.imread(img_file)
            img = cv2.imdecode(np.fromstring(img_file, np.int8), 1)
            # img = img_file
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)

            img = cv2.resize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            # images[i] = img
            self.imgs = list()
            self.imgs.append(img)  # CHeck image input tensor and prepare for one image
            self.cpi = 1
            self.transform = transform
            self.dataset_size = len(self.imgs)

        else:
            self.split = split
            assert self.split in {'TRAIN', 'VAL', 'TEST'}

            # Open hdf5 file where images are stored
            self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
            self.imgs = self.h['images']

            # Captions per image
            self.cpi = self.h.attrs['captions_per_image']

            # Load encoded captions (completely into memory)
            with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
                self.captions = json.load(j)

            # Load caption lengths (completely into memory)
            with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
                self.caplens = json.load(j)

            # load style factors
            with open(os.path.join(data_folder, self.split + '_STYLES_' + data_name + '.json'), 'r') as j:
                self.styles = json.load(j)

            # load image id
            with open(os.path.join(data_folder, self.split + '_ID_' + data_name + '.json'), 'r') as j:
                self.ids = json.load(j)

            # PyTorch transformation pipeline for the image (normalizing, etc.)
            self.transform = transform

            # Total number of datapoints
            self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        if self.inference:
            img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
            if self.transform is not None:
                img = self.transform(img)
            return img, '', '', '', '', i

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        # load in style factors
        len_class = self.styles[i]['length_class']
        sentence_length = torch.LongTensor([len_class])

        img_id = self.ids[i]

        if self.split is 'TRAIN':
            return img, caption, caplen, caption, sentence_length, img_id
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, sentence_length, img_id

    def __len__(self):
        return self.dataset_size
