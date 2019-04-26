# Example of the output format for evaluation for 3d semantic label and instance prediction.
# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# python imports
import math
import os, sys, argparse
import inspect
import json
import csv
import pickle

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

import utils

def save_scannet_benchmark(instance_class, instance_mask, instance_conf, verts_len, output_dir, scene_id):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predicted_masks'), exist_ok=True)
    f = open(os.path.join(output_dir, scene_id + '.txt'), 'w')
    for instance_id in instance_class:
        cls = instance_class[instance_id]
        score = instance_conf[instance_id]
        mask_file = 'predicted_masks/' + scene_id + '_' + '{:03d}'.format(instance_id) + '.txt'
        f.write(mask_file + ' ' + str(cls) + ' ' + str(float(score)) +'\n')
        mask = np.zeros(verts_len).astype(np.uint8)
        mask[instance_mask[instance_id]] = 1
        np.savetxt(os.path.join(output_dir, mask_file), mask, fmt='%u')

    f.close()

    print('done!')

def load_pred(pred_folder):
    scene = np.zeros((400, 200, 400))
    pred_box = np.load(os.path.join(pred_folder, 'pred_box.npy'))[:,:6]
    pred_class = np.load(os.path.join(pred_folder, 'pred_class.npy'))
    pred_conf = np.load(os.path.join(pred_folder, 'pred_conf.npy'))
    pred_mask = pickle.load(open(os.path.join(pred_folder, 'pred_mask'), 'rb'))
    sort_index = pickle.load(open(os.path.join(pred_folder, 'pred_mask_index'), 'rb'))

    pred_box = pred_box[sort_index]
    pred_conf = pred_conf[sort_index]
    pred_class = pred_class[sort_index]

    #pred_mask = pickle.load(open(os.path.join(pred_folder, 'gt_mask'), 'rb'))
    #pred_box = np.load(os.path.join(pred_folder, 'gt_box.npy'))[:,:6]
    #pred_class = np.load(os.path.join(pred_folder, 'gt_class.npy'))
    #pred_conf = np.zeros_like(pred_class)

    for box_ind, box in enumerate(pred_box):
        minx = int(round(box[0]))
        miny = int(round(box[1]))
        minz = int(round(box[2]))

        maxx = int(round(box[3]))
        maxy = int(round(box[4]))
        maxz = int(round(box[5]))
        for i in range(minx, maxx):
            for j in range(miny, maxy):
                for k in range(minz, maxz):
                    if pred_mask[box_ind][i-minx,j-miny,k-minz] !=0 and scene[i,j,k] == 0:
                        scene[i,j,k] =  box_ind * 100 + pred_class[box_ind] + pred_conf[box_ind] - 0.01
    return scene

def nn_search(scene, x, y, z):
    if scene[x, y, z] != 0:
        return x, y, z
    else:
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if scene[x+i, y+j, z+k] != 0:
                        return x+i,y+j,z+k

    return -1,-1,-1

def export(mesh_vertices, world2grid, scene, output_dir, scene_id):
    instance_mask = {}
    instance_conf = {}
    instance_class = {}
    for ind, vertex in enumerate(mesh_vertices):
        grid_coord = np.round(np.matmul(world2grid, np.append(vertex, 1)))
        coord_3d_x = int(round(grid_coord[0])) 
        coord_3d_y = int(round(grid_coord[1])) 
        coord_3d_z = int(round(grid_coord[2])) 
        coord_3d_x, coord_3d_y, coord_3d_z = nn_search(scene, coord_3d_x, coord_3d_y, coord_3d_z)
        if coord_3d_x == -1:
            continue

        conf = np.modf(scene[coord_3d_x, coord_3d_y, coord_3d_z])[0]
        instance_id = int(int(scene[coord_3d_x, coord_3d_y, coord_3d_z]) / 100)
        class_id = int(scene[coord_3d_x, coord_3d_y, coord_3d_z]) % 100
        if instance_id not in instance_class:
            instance_class[instance_id] = class_id
            instance_mask[instance_id] = [ind]
            instance_conf[instance_id] = conf
        else:
            instance_mask[instance_id].append(ind)

    save_scannet_benchmark(instance_class, instance_mask, instance_conf, len(mesh_vertices), output_dir, scene_id)
            
def load_matrix(filename):
    padding = [10, 16, 10, 0]
    matrix = np.zeros((4, 4))
    with open(filename) as f:
        lines = f.readlines()
    for ind, line in enumerate(lines):
        matrix[ind][0] = float(line.split()[0])
        matrix[ind][1] = float(line.split()[1])
        matrix[ind][2] = float(line.split()[2])
        matrix[ind][3] = float(line.split()[3]) - padding[ind]
    return  matrix

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', default='/mnt/raid/ji/3D-SIS/ours/results/ScanNet/benchmark/test')
parser.add_argument('--output_dir', default="./ScanNet_Benchmark_Result", help='output file')
parser.add_argument('--scan_path', default='/mnt/canis_datasets/ScanNet/public/v2/scans')
parser.add_argument('--frames', default='/mnt/local_datasets/ScanNet/frames_square')

opt = parser.parse_args()
def main():
    for ind, pred_folder in enumerate(os.listdir(opt.pred_dir)):
        print('{}/{}'.format(ind, len(os.listdir(opt.pred_dir))))
        mesh_file = os.path.join(opt.scan_path, pred_folder, pred_folder + '_vh_clean_2.ply')

        if not os.path.isdir(os.path.join(opt.pred_dir, pred_folder)):
            continue
        if not os.path.exists(mesh_file):
            continue

        os.makedirs(opt.output_dir, exist_ok=True)

        world2grid = load_matrix(os.path.join(opt.frames, pred_folder, 'world2grid.txt'))
        mesh_vertices = utils.read_mesh_vertices(mesh_file)
        scene = load_pred(os.path.join(opt.pred_dir, pred_folder))
        export(mesh_vertices, world2grid, scene, opt.output_dir, pred_folder)

if __name__ == '__main__':
    main()
