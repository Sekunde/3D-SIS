import os
import numpy as np
from plyfile import PlyData, PlyElement
import sys
sys.path.append('.')
from tools.visualization import write_ply
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

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default="./ScanNet_Benchmark_Result")
parser.add_argument('--output_dir', default="./ScanNet_Benchmark_Result_Visualization")
parser.add_argument('--scan_path', default='/mnt/canis_datasets/ScanNet/public/v2/scans')

opt = parser.parse_args()

def main():
    res_folder = opt.result_dir
    ply_folder = opt.scan_path
    output_dir = opt.output_dir

    reader_ins = Benchmark_reader(res_folder)
    for folder in os.listdir(res_folder):
        print(folder)
        # ply reader
        ply_file = os.path.join(ply_folder, folder.split('.')[0], folder.split('.')[0]+'_vh_clean_2.ply')
        ply_data = PlyData.read(ply_file)
        points = []
        for point in ply_data.elements[0].data:
            points.append([point[0], point[1], point[2]])
        points = np.array(points)
        colors = np.zeros_like(points)

        # instance reader
        instances = reader_ins[folder]
        for instance_idx, instance_key in enumerate(instances.keys()):
            r, g, b = create_color_palette()[int((instance_idx + 1)%41)]
            colors[instances[instance_key]['points'].nonzero()[0].astype(np.int32)] = [r,g,b]

        output_file = os.path.join(output_dir, folder.split('.')[0] + '.ply')
        write_ply(points, colors, None, output_file)

if __name__ == '__main__':
    main()
