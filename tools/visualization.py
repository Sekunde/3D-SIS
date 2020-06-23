"""
    Simple Usage example (with 3 images)
"""
import os
import math 
import numpy as np
import argparse
import pickle
import sys
sys.path.append('.')
from lib.datasets.BinaryReader import BinaryReader
from plyfile import PlyData, PlyElement

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


def read_ply(ply_file):
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)

    points = []
    colors = []
    indices = []
    for x,y,z,r,g,b,a in ply_data['vertex']:
        points.append([x,y,z])
        colors.append([r,g,b])
    for face in ply_data['face']:
        indices.append([face[0][0], face[0][1], face[0][2]])
    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)
    return points, indices, colors

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(bbox, output_file=None):
    """
    bbox: np array (n, 7), last one is instance/label id
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.02
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for box in bbox:
        box_min = np.array([box[0], box[1], box[2]])
        box_max = np.array([box[3], box[4], box[5]])
        r, g, b = create_color_palette()[int(box[6]%41)]
        edges = get_bbox_edges(box_min, box_max)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[r/255.0,g/255.0,b/255.0] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)

    if output_file is None:
        return verts, colors, indices

    write_ply(verts, colors, indices, output_file)

def write_mask_pointcloud(mask, output_file):
    """
    mask: numpy array (x,y,z), in which instance/label id
    output_file: string

    """
    def make_voxel_mesh(box_min, box_max, color): 
        vertices = [
            np.array([box_max[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_min[1], box_min[2]])
        ]
        return vertices

    scale = 1
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for z in range(mask.shape[2]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]):
                if mask[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) - 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts = make_voxel_mesh(box_min, box_max, np.array(create_color_palette()[int(mask[x,y,z]%41)])/255.0)
                    verts.extend(box_verts)
    write_ply(verts, None, None, output_file)

def write_mask(mask, output_file):
    """
    mask: numpy array (x,y,z), in which instance/label id
    output_file: string

    """
    def make_voxel_mesh(box_min, box_max, color): 
        vertices = [
            np.array([box_max[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_min[1], box_min[2]])
        ]

        colors = [
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]])
        ]
        indices = [
            np.array([1, 2, 3], dtype=np.uint32),
            np.array([1, 3, 0], dtype=np.uint32),
            np.array([0, 3, 7], dtype=np.uint32),
            np.array([0, 7, 4], dtype=np.uint32),
            np.array([3, 2, 6], dtype=np.uint32),
            np.array([3, 6, 7], dtype=np.uint32),
            np.array([1, 6, 2], dtype=np.uint32),
            np.array([1, 5, 6], dtype=np.uint32),
            np.array([0, 5, 1], dtype=np.uint32),
            np.array([0, 4, 5], dtype=np.uint32),
            np.array([6, 5, 4], dtype=np.uint32),
            np.array([6, 4, 7], dtype=np.uint32)
        ]
        return vertices, colors, indices

    scale = 1
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for z in range(mask.shape[2]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]):
                if mask[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) - 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts, box_color, box_ind = make_voxel_mesh(box_min, box_max, np.array(create_color_palette()[int(mask[x,y,z]%41)])/255.0)
                    cur_num_verts = len(verts)
                    box_ind = [x + cur_num_verts for x in box_ind]
                    verts.extend(box_verts)
                    indices.extend(box_ind)
                    colors.extend(box_color)
    write_ply(verts, colors, indices, output_file)

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='3D-SIS')
    parser.add_argument('--path', type=str, default='../results/')
    parser.add_argument('--mode', type=str, default='npy')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.mode == 'data':
        path = args.path
        for chunk in os.listdir(path):
            reader = BinaryReader(os.path.join(path, chunk))

            # data
            dimX, dimY, dimZ = reader.read('UINT64', 3)
            data = reader.read('float', dimX * dimY * dimZ)
            data = np.reshape(data, (dimX, dimY, dimZ), order='F').astype(np.float32)

            data = np.where(abs(data) <= 1.0, 1, 0)
            write_mask(data, '{}_data.ply'.format(chunk))
            
            # bbox
            (num_box,) = reader.read('uint32')
            gt_box = np.zeros((num_box, 7), dtype=np.float32)
            for i in range(num_box):
                    minx, miny, minz, maxx, maxy, maxz = reader.read('float', 6)
                    (labelid,) = reader.read('uint32')
                    gt_box[i] = [math.floor(minx), math.floor(miny), math.floor(minz), math.ceil(maxx), math.ceil(maxy), math.ceil(maxz), labelid]
            

            # mask
            (num_mask,) = reader.read('uint32')
            gt_mask = np.zeros((dimX+100, dimY+100, dimZ+100))
            for i in range(num_mask):
                (labelid,) = reader.read('uint32')
                dimX, dimY, dimZ = reader.read('UINT64', 3)
                mask_data = reader.read('uint16', dimX * dimY * dimZ)
                mask_data = np.reshape(mask_data, (dimX, dimY, dimZ), order='F').astype(np.float32)
                for mask_i in range(int(gt_box[i][0]), int(gt_box[i][3])):
                    for mask_j in range(int(gt_box[i][1]), int(gt_box[i][4])):
                        for mask_k in range(int(gt_box[i][2]), int(gt_box[i][5])):
                            if mask_i < 0 or mask_j < 0 or mask_k < 0:
                                continue
                            if gt_mask[mask_i, mask_j, mask_k] == 0 and mask_data[mask_i - int(gt_box[i][0]), mask_j - int(gt_box[i][1]), mask_k - int(gt_box[i][2])] > 0:
                                gt_mask[mask_i, mask_j, mask_k] =  mask_data[mask_i - int(gt_box[i][0]), mask_j - int(gt_box[i][1]), mask_k - int(gt_box[i][2])]

            # part in vol
            (num_box,) = reader.read('uint32')
            box_stats = []
            for i in range(num_box):
                (part_in_vol,) = reader.read('float')
                box_stats.append(part_in_vol)

            write_bbox(gt_box, '{}_box.ply'.format(chunk))
            write_mask(gt_mask, '{}_mask.ply'.format(chunk))

            # image
            world2grid = reader.read('float', 16)
            print('world2grid matrix:')
            print(world2grid)
            (num_images,) = reader.read('uint32')
            print('assocated image index in this chunk:')
            for i in range(num_images):
                (frameid,) = reader.read('uint32')  
                print(frameid)
            reader.close()

    elif args.mode == 'result':
        # visualize results
        path = args.path
        for scene in os.listdir(path):
            scene_data = np.load(os.path.join(path, scene, 'scene.npy'))
            write_mask(scene_data, os.path.join(path, scene, 'scene.ply'))

            #-------------pred------------------
            try:
                pred_roi = np.load(os.path.join(path, scene, 'pred_roi.npy'))[:,:6]
                pred_roi_exist_flag = True
            except:
                pred_roi_exist_flag = False


            try:
                pred_bbox = np.load(os.path.join(path, scene, 'pred_box.npy'))[:,:6]
                pred_bbox_exist_flag = True
            except:
                pred_bbox_exist_flag = False

            try:
                pred_class = np.load(os.path.join(path, scene, 'pred_class.npy'))
                pred_class_exist_flag = True
            except:
                pred_class_exist_flag = False

            try:
                pred_mask = pickle.load(open(os.path.join(path, scene, 'pred_mask'), 'rb'))
                sort_index = pickle.load(open(os.path.join(path, scene, 'pred_mask_index'), 'rb'))
                pred_bbox = pred_bbox[sort_index]
                pred_class = pred_class[sort_index]
                pred_mask_exist_flag = True
            except:
                pred_mask_exist_flag = False

            if pred_roi_exist_flag:
                write_bbox(np.concatenate([pred_roi, np.ones_like(pred_roi)[:,0:1]], 1), os.path.join(path, scene, 'pred_roi.ply'))

            if pred_bbox_exist_flag:
                if pred_class_exist_flag:
                    write_bbox(np.concatenate([pred_bbox, np.expand_dims(pred_class,1)], 1), os.path.join(path, scene, 'pred_bbox.ply'))
                else:
                    write_bbox(np.concatenate([pred_bbox, np.oneos_like(pred_bbox)[:,0:1]], 1), os.path.join(path, scene, 'pred_bbox.ply'))

            if pred_mask_exist_flag:
                mask_scene = np.zeros_like(scene_data)
                for box_ind, box in enumerate(pred_bbox):
                    minx = int(round(box[0]))
                    miny = int(round(box[1]))
                    minz = int(round(box[2]))

                    maxx = int(round(box[3]))
                    maxy = int(round(box[4]))
                    maxz = int(round(box[5]))
                    for i in range(minx, maxx):
                        for j in range(miny, maxy):
                            for k in range(minz, maxz):
                                if pred_mask[box_ind][i-minx,j-miny,k-minz] !=0 and mask_scene[i,j,k] == 0:
                                    mask_scene[i,j,k] =  pred_class[box_ind] if pred_class_exist_flag else 1
                write_mask(mask_scene, os.path.join(path, scene, 'pred_mask.ply'))


            #---------------gt--------------------
            try:
                gt_bbox = np.load(os.path.join(path, scene, 'gt_box.npy'))[:,:6]
                gt_bbox_exist = True
            except:
                gt_bbox_exist = False

            try:
                gt_class = np.load(os.path.join(path, scene, 'gt_class.npy'))
                gt_class_exist_flag = True
            except:
                gt_class_exist_flag = False

            try:
                gt_mask = pickle.load(open(os.path.join(path, scene, 'gt_mask'), 'rb'))
                gt_mask_exist_flag = True
            except:
                gt_mask_exist_flag = False


            if gt_bbox_exist:
                if gt_class_exist_flag:
                    write_bbox(np.concatenate([gt_bbox, np.expand_dims(gt_class,1)], 1), os.path.join(path, scene, 'gt_bbox.ply'))
                else:
                    write_bbox(np.concatenate([gt_bbox, np.ones_like(gt_bbox)[:,0:1]], 1), os.path.join(path, scene, 'gt_bbox.ply'))

            if gt_mask_exist_flag:
                mask_scene = np.zeros_like(scene_data)
                for box_ind, box in enumerate(gt_bbox):
                    minx = int(round(box[0]))
                    miny = int(round(box[1]))
                    minz = int(round(box[2]))

                    maxx = int(round(box[3]))
                    maxy = int(round(box[4]))
                    maxz = int(round(box[5]))
                    for i in range(minx, maxx):
                        for j in range(miny, maxy):
                            for k in range(minz, maxz):
                                if gt_mask[box_ind][i-minx,j-miny,k-minz] !=0 and mask_scene[i,j,k] == 0:
                                    mask_scene[i,j,k] =  gt_class[box_ind] if gt_class_exist_flag else 1
                write_mask(mask_scene, os.path.join(path, scene, 'gt_mask.ply'))


