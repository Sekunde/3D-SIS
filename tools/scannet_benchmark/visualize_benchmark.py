import os
import numpy as np
import torch
import sys
sys.path.append('.')
from tools.visualization import write_ply, read_ply, write_bbox
import argparse

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


class Benchmark_reader(object):
    def __init__(self, res_path):
        self.res_path = res_path
        self.res_txt = os.listdir(res_path)
    def __getitem__(self, txt_file):
        if txt_file  not in self.res_txt:
            print('{} not exist'.format(txt_file))
            return 0
        else:
            ret_instances = {}
            instances = open(os.path.join(self.res_path, txt_file)).readlines()
            for idx, instance in enumerate(instances):
                label = int(instance.split()[1])
                point_index = np.loadtxt(os.path.join(self.res_path, instance.split()[0]))
                ret_instances[idx] = {}
                ret_instances[idx]['points'] = point_index
                ret_instances[idx]['label'] = label
            return ret_instances

def coords_multiplication(A, B):
    '''
    A: 4x4
    B: nx3
    '''
    
    if isinstance(A, torch.Tensor):
        device = torch.device("cuda:0" if A.get_device() != -1 else "cpu") 
        B = torch.cat([B.t(), torch.ones((1, B.shape[0]), device=device)])
        return torch.mm(A, B).t()[:,:3]
    elif isinstance(A, np.ndarray):
        B = np.concatenate([np.transpose(B), np.ones((1, B.shape[0]))])
        return np.transpose(np.dot(A, B))[:,:3]

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default="./ScanNet_Benchmark_Result")
parser.add_argument('--output_dir', default="./ScanNet_Benchmark_Result_Visualization")
parser.add_argument('--scan_path', default='/mnt/canis_datasets/ScanNet/public/v2/scans')

opt = parser.parse_args()

def main():
    res_folder = opt.result_dir
    ply_folder = opt.scan_path
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)

    reader_ins = Benchmark_reader(res_folder)
    for folder in os.listdir(res_folder):
        if os.path.isdir(os.path.join(res_folder, folder)):
            continue
        print(folder)

        # ply reader
        ply_file = os.path.join(ply_folder, folder.split('.')[0], folder.split('.')[0]+'_vh_clean_2.ply')
        alignment_file = os.path.join(ply_folder, folder.split('.')[0], folder.split('.')[0]+'.txt')
        alignment = open(alignment_file).readlines()[0]
        alignment = np.array([float(a_) for a_ in alignment.split()[2:]]).reshape(4, 4)
        points, faces, _ = read_ply(ply_file)
        colors = np.zeros_like(points) + np.array([64, 64, 96])
        points = coords_multiplication(alignment, points)

        # instance reader
        bbox_points = []
        bbox_faces = []
        bbox_colors = []
        previous_ = 0
        instances = reader_ins[folder]
        for instance_idx, instance_key in enumerate(instances.keys()):
            r, g, b = create_color_palette()[int((instance_idx + 1)%41)]

            instance_points = points[instances[instance_key]['points'].nonzero()[0].astype(np.int32)]

            # get bbox mesh
            minx, miny, minz = np.min(instance_points, 0)
            maxx, maxy, maxz = np.max(instance_points, 0)
            ins_verts, ins_colors, ins_faces = write_bbox([[minx, miny, minz, maxx, maxy, maxz, int(instance_idx + 1)]], None)
            bbox_points.extend(np.array(ins_verts))
            bbox_faces.extend(np.array(ins_faces)+previous_)
            bbox_colors.extend(np.array(ins_colors))
            previous_ += len(ins_verts)

            colors[instances[instance_key]['points'].nonzero()[0].astype(np.int32)] = [r/255.0,g/255.0,b/255.0]

        output_file = os.path.join(output_dir, folder.split('.')[0] + '.ply')

        bbox_points = np.array(bbox_points)
        bbox_colors = np.array(bbox_colors)
        bbox_faces = np.array(bbox_faces) + points.shape[0]
        write_ply(np.concatenate([points, bbox_points]), np.concatenate([colors, bbox_colors]), np.concatenate([faces, bbox_faces]), output_file)

if __name__ == '__main__':
    main()
