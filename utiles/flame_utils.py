import os
import torch
import numpy as np


def get_mesh(flamelayer, data):

    expression_params = torch.tensor(data['expression'], dtype=torch.float32).cuda()  # expression
    shape_params = torch.zeros(8, 100).cuda()  # identity
    pose_params = torch.tensor(data['pose'], dtype=torch.float32).cuda()  # pose
    pose_params = torch.cat([torch.zeros(pose_params.shape[0], 3).cuda(), pose_params[:, 3:]], dim=1)

    current_length = expression_params.shape[0]
    desired_length = ((current_length - 1) // 8 + 1) * 8
    padding_length = desired_length - current_length
    expression_params = torch.nn.functional.pad(expression_params, (0, 0, 0, padding_length))  # 将expression_params的长度补齐为8的倍数
    pose_params = torch.nn.functional.pad(pose_params, (0, 0, 0, padding_length))  # 将pose_params的长度补齐为8的倍数

    vertices = []
    for i in range(0, desired_length, 8):
        current_expression = expression_params[i:i+8]
        current_pose = pose_params[i:i+8]
        vertice, _ = flamelayer(shape_params, current_expression, current_pose)
        vertices.append(vertice)

    vertices = torch.cat(vertices, dim=0)[:current_length]
        
    return vertices

def torch2mesh(flamelayer, expression_params, pose_params):
    expression_params = expression_params.squeeze(0)
    pose_params = pose_params.squeeze(0)
    shape_params = torch.zeros(8, 100).to(expression_params.device)  # identity
    current_length = expression_params.shape[0]
    desired_length = ((current_length - 1) // 8 + 1) * 8
    padding_length = desired_length - current_length
    expression_params = torch.nn.functional.pad(expression_params, (0, 0, 0, padding_length))  # 将expression_params的长度补齐为8的倍数
    pose_params = torch.nn.functional.pad(pose_params, (0, 0, 0, padding_length))  # 将pose_params的长度补齐为8的倍数

    vertices = []
    for i in range(0, desired_length, 8):
        current_expression = expression_params[i:i+8]
        current_pose = pose_params[i:i+8]
        vertice, _ = flamelayer(shape_params, current_expression, current_pose)
        vertices.append(vertice)

    vertices = torch.cat(vertices, dim=0)[:current_length]
    vertices = torch.round(vertices, decimals=4)  # 将motion保留三位小数，以防止数据过长。
    vertices = vertices.flatten(-2).unsqueeze(0)
    return vertices