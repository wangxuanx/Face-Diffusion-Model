"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import os
import numpy as np
import torch
from FLAME import FLAME
from config import get_config

def test():
    config = get_config()
    radian = np.pi/180.0
    flamelayer = FLAME(config)

    # Creating a batch of mean shapes
    shape_params = torch.zeros(8, 100).cuda()

    # Creating a batch of different global poses
    # pose_params_numpy[:, :3] : global rotaation
    # pose_params_numpy[:, 3:] : jaw rotaation
    pose_params_numpy = np.array([[0.0, 30.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, -30.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 85.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, -48.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 10.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, -15.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0*radian, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, -0.0*radian, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(8, 50, dtype=torch.float32).cuda()
    flamelayer.cuda()

    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
    vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project
    print(vertice.size(), landmark.size())

    if config.optimize_eyeballpose and config.optimize_neckpose:
        neck_pose = torch.zeros(8, 3).cuda()
        eye_pose = torch.zeros(8, 6).cuda()
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    # Visualize Landmarks
    # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
    # faces = flamelayer.faces
    # for i in range(8):
    #     vertices = vertice[i].detach().cpu().numpy().squeeze()
    #     joints = landmark[i].detach().cpu().numpy().squeeze()
    #     vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    #     tri_mesh = trimesh.Trimesh(vertices, faces,
    #                                 vertex_colors=vertex_colors)
    #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    #     scene = pyrender.Scene()
    #     scene.add(mesh)
    #     sm = trimesh.creation.uv_sphere(radius=0.005)
    #     sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    #     tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    #     tfs[:, :3, 3] = joints
    #     joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    #     scene.add(joints_pcl)
    #     pyrender.Viewer(scene, use_raymond_lighting=True)

def main():
    config = get_config()
    flamelayer = FLAME(config).cuda()

    path = '/data/WX/fdm/M003_WX'
    save_path = '/data/WX/fdm/M003_WX_mesh'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    npz_list = os.listdir(path)
    for npz in npz_list:
        if npz[-4:] != '.npz':
            continue
        print(npz)
        data = np.load(os.path.join(path, npz))
        expression_params = torch.tensor(data['expression'], dtype=torch.float32).cuda()  # expression
        shape_params = torch.zeros(8, 100).cuda()  # identity
        pose_params = torch.tensor(data['pose'], dtype=torch.float32).cuda()  # pose
        pose_params = torch.cat([torch.zeros(pose_params.shape[0], 3).cuda(), pose_params[:, 3:]], dim=1)

        current_length = expression_params.shape[0]
        desired_length = ((current_length - 1) // 8 + 1) * 8
        padding_length = desired_length - current_length
        expression_params = torch.nn.functional.pad(expression_params, (0, 0, 0, padding_length))
        pose_params = torch.nn.functional.pad(pose_params, (0, 0, 0, padding_length))

        vertices = []
        for i in range(0, desired_length, 8):
            current_expression = expression_params[i:i+8]
            current_pose = pose_params[i:i+8]
            vertice, landmark = flamelayer(shape_params, current_expression, current_pose)
            vertices.append(vertice)

        vertices = torch.cat(vertices, dim=0)[:current_length]
        vertices = vertices.detach().cpu().numpy()
        np.save(os.path.join(save_path, npz[:-4]), vertices)

if __name__ == '__main__':
    main()