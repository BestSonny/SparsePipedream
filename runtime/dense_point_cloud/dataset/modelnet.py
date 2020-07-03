from typing import Callable, Iterable, Optional, Union, List

import torch
import os
from glob import glob
from tqdm import tqdm

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs
import kaolin.transforms as tfs
from torch.utils.data import Dataset

class ModelNet(object):
    def __init__(self, basedir: str,
                 split: Optional[str] = 'train',
                 categories: Optional[Iterable] = ['bed'],
                 transform: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        assert split.lower() in ['train', 'test']

        self.basedir = basedir
        self.transform = transform
        self.device = device
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(basedir):
            raise ValueError('ModelNet was not found at "{0}".'.format(basedir))

        available_categories = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]

        for cat_idx, category in enumerate(categories):
            assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
                category, available_categories)

            cat_paths = glob(os.path.join(basedir, category, split.lower(), '*.off'))

            self.cat_idxs += [cat_idx] * len(cat_paths)
            self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
            self.filepaths += cat_paths

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = TriangleMesh.from_off(self.filepaths[index])
        data.to(self.device)
        if self.transform:
            data = self.transform(data)

        return data

class ModelNetVoxels(object):
    def __init__(self, basedir: str, cache_dir: Optional[str] = None, 
                 split: Optional[str] = 'train',
                 resolution: int = 32,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        self.basedir = basedir
        self.device = torch.device(device)
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(basedir, 'cache')
        self.resolution = resolution
        self.cache_transforms = {}

        categories = ['sofa', 'cup', 'plant', 'radio',
                      'sink', 'bookshelf', 'toilet', 'lamp', 
                      'guitar', 'dresser', 'laptop', 'wardrobe', 
                      'flower_pot', 'piano', 'xbox', 'glass_box', 
                      'bottle', 'stairs', 'table', 'bench', 
                      'tv_stand', 'bathtub', 'stool', 'person', 
                      'chair', 'car', 'keyboard', 'night_stand', 
                      'mantel', 'airplane', 'monitor', 'bed', 
                      'tent', 'vase', 'desk', 'bowl', 
                      'door', 'cone', 'curtain', 'range_hood']

        mesh_dataset = ModelNet(basedir=basedir, split=split, categories=categories, device=device)

        self.names = mesh_dataset.names
        self.categories = mesh_dataset.categories
        self.cat_idxs = mesh_dataset.cat_idxs

        
        self.cache_transforms[self.resolution] = tfs.CacheCompose([
            tfs.TriangleMeshToVoxelGrid(self.resolution, normalize=True, vertex_offset=0.5),
            tfs.FillVoxelGrid(thresh=0.5),
            tfs.ExtractProjectOdmsFromVoxelGrid()
        ], self.cache_dir)

        desc = 'converting to voxels to resolution {0}'.format(self.resolution)
        for idx in tqdm(range(len(mesh_dataset)), desc=desc, disable=False):
            name = mesh_dataset.names[idx]
            if name not in self.cache_transforms[self.resolution].cached_ids:
                mesh = mesh_dataset[idx]
                mesh.to(device=device)
                self.cache_transforms[self.resolution](name, mesh)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        voxel = self.cache_transforms[self.resolution](name)
        category = torch.tensor(self.cat_idxs[index], dtype=torch.long, device=self.device)
        return voxel.unsqueeze_(0), category
        