import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils.config import cfg


def collate_fn(batch):
    """

    :param batch: list of dicts
    :return: dict
    """

    nearest_images = {}
    if cfg.USE_IMAGES:
        depths = []
        poses = []
        world2grid = []
        for b in batch:
            x = b['nearest_images']
            num_images = len(x['depths'])
            max_num_images = cfg.NUM_IMAGES if not cfg.RANDOM_NUM_IMAGES else np.random.randint(low=1,high=cfg.NUM_IMAGES+1)  # [low,high)
            if max_num_images < num_images and cfg.MODE == 'train':
                num_images = max_num_images
                x['images'] = x['images'][:num_images]
                x['depths'] = x['depths'][:num_images]
                x['poses'] = x['poses'][:num_images]
            depths.append(torch.from_numpy(np.array(x['depths'])))
            poses.append(torch.from_numpy(np.array(x['poses'])))
            world2grid.append(torch.from_numpy(x['world2grid']).expand(num_images, 4, 4))

        nearest_images = {
            'images': [torch.from_numpy(np.stack(x['nearest_images']['images'], 0).astype(np.float32)) for x in batch],
            'depths': depths, 'poses': poses, 'world2grid': world2grid
        }
    return {
        'id': [x['id'] for x in batch],
        'data': torch.stack([torch.from_numpy(x['data']) for x in batch], 0),
        'gt_box': [torch.from_numpy(x['gt_box']) for x in batch if x['gt_box'].shape[0]!=0],
        'gt_mask': [[torch.from_numpy(y) for y in x['gt_mask']] for x in batch if len(x['gt_mask'])!=0],
        'nearest_images': nearest_images,
        'image_files': batch[0]['image_files']
    }


def get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=1):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                             shuffle=shuffle, num_workers=num_workers)
    return data_loader
