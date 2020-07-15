import os
import os.path
import torch.utils.data as data
import torch
import numpy as np
import sys
import json
import MinkowskiEngine as ME
import time
from typing import Callable, Iterable, Optional, Union, List
from glob import glob
from tqdm import tqdm

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs
import kaolin.transforms as tfs
from torch.utils.data import Dataset

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 shared_dict={},
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 voxel_size = 0.05):
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.voxel_size = voxel_size
        self.data_time = AverageMeter()
        self.data_agu_time = AverageMeter()
        self.voxel_time = AverageMeter()
        self.cache = shared_dict

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append(
                    (os.path.join(self.root, category, 'points', uuid +'.pts'),
                     os.path.join(self.root, category, 'points_label', uuid +'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print("classes:", self.classes)
        with open('misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print("seg_classes:", self.seg_classes, self.num_seg_classes)
        
    def __getitem__(self, index):
        start = time.time()
        if index in self.cache:
            point_set, cls = self.cache[index]
            #print("Cached index:", index)
        else:
            fn = self.datapath[index] 
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            self.cache[index] = (point_set, cls)
            #print("New index:", index, "cache length:", len(self.cache))
        self.data_time.update(time.time() - start)

        start = time.time()
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)) ,0)
        point_set = point_set / dist  # scale
 
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)] ,[np.sin(theta), np.cos(theta)]])
            point_set[: ,[0 ,2]] = point_set[: ,[0 ,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set) 
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        self.data_agu_time.update(time.time() - start)

        start = time.time()
        quantized_coords = np.floor(point_set / self.voxel_size)
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
        self.voxel_time.update(time.time() - start)
        #print("index:", index, self.data_time.avg, self.data_agu_time.avg, self.voxel_time.avg)

        if self.classification:
            return quantized_coords[inds], quantized_coords[inds], cls
        else:
            seg = np.loadtxt(fn[2]).astype(np.int64)
            seg = torch.from_numpy(seg)
            return quantized_coords[inds], quantized_coords[inds], seg[inds]

    def __len__(self):
        return len(self.datapath)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_pointcloud_fn(list_data):
    start = time.time()
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    coords_batch, feats_batch, quantized_labels = ME.utils.sparse_collate(coords, feats, labels)

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': quantized_labels,
    }

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

class ModelNetMinkowski(object):
    def __init__(self, basedir: str, cache_dir: Optional[str] = None, 
                 split: Optional[str] = 'train',
                 total_point: int = 16384,
                 num_points: int = 4096,
                 voxel_size: float = 0.02,
                 sample_ratio: float = 1.0,
                 device: Optional[Union[torch.device, str]] = 'cpu',
                 repeat: Optional[int] = 1):

        self.basedir = basedir
        self.device = torch.device(device)
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(basedir, 'cache_points')
        self.num_points = num_points
        self.total_point = total_point
        self.voxel_size = voxel_size
        self.training = split.lower()
        self.repeat = repeat
        self.sample_ratio = sample_ratio
        print("voxel_size:", voxel_size, "num_points:", num_points, "sample_ratio:", sample_ratio)

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

    
        self.cache_transforms = tfs.CacheCompose([
            tfs.TriangleMeshToPointCloud(num_samples=total_point),
            tfs.NormalizePointCloud(),
        ], self.cache_dir)

        for idx in tqdm(range(len(mesh_dataset)), disable=False):
            name = mesh_dataset.names[idx]
            # if name == 'cone_0117' or name == 'curtain_0066':
            if name not in self.cache_transforms.cached_ids:
                mesh = mesh_dataset[idx]
                mesh.to(device=device)
                self.cache_transforms(name, mesh)


    def __len__(self):
        if self.training == 'train':
            return len(self.names)*self.repeat
        else:
            return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        index = index % len(self.names)
        name = self.names[index]
        point_clouds = self.cache_transforms(name)
        point_clouds = point_clouds - point_clouds.min(dim=0)[0]
        scale = point_clouds.max(dim=0)[0] - point_clouds.min(dim=0)[0]
        point_clouds = (point_clouds -point_clouds.min(dim=0)[0])/(scale*1.0)
        if self.training == 'train':
            choice = np.random.choice(point_clouds.shape[0], int(self.num_points*self.sample_ratio), replace=False)
        else:
            choice = np.random.choice(point_clouds.shape[0], self.num_points, replace=False)
        point_clouds = point_clouds[choice]
        quantized_coords = point_clouds.div(self.voxel_size).floor()
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
        feats = np.empty([quantized_coords[inds].size(0), 1])
        feats.fill(1)
        feats = torch.from_numpy(feats.astype(np.float32))
        category = torch.tensor([self.cat_idxs[index]], dtype=torch.long, device=self.device)
        return quantized_coords[inds], feats, category
        


if __name__ == '__main__':
    train_dataset = ModelNetMinkowski(basedir="../../dense_point_cloud/ModelNet40",
                                      split='train',
                                      voxel_size=0.05)
    train_dataset = ModelNetMinkowski(basedir="../../dense_point_cloud/ModelNet40",
                                      split='test',
                                      voxel_size=0.05)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=16,
                                        pin_memory=True,
                                        collate_fn=collate_pointcloud_fn)
    print("len(train_dataset):", len(train_dataset))
    print("len(train_loader)", len(train_loader))
    data_time = AverageMeter()
    for i in range(10):
        start_epoch = time.time()
        start = time.time()
        for j, data_dict in enumerate(train_loader):
            coords = data_dict['coords']
            feats = data_dict['feats']
            labels = data_dict['labels']
            sin = ME.SparseTensor(
            feats, #coords[:, :3] * args.voxel_size,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
            )  #.to(device)
            sin = sin.to('cuda')
            labels = labels.to('cuda')
            torch.cuda.synchronize()
            data_time.update(time.time() - start)
            #print("Read data time:", i, j, data_time.val, data_time.avg)
            start = time.time()
        print("Epoch load time:", time.time() - start_epoch, data_time.avg)
