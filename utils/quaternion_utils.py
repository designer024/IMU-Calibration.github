import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

import utils.mysulplotter as plotter
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
plt3D = plotter.Plotter3D(usebuffer=False, no_margin=True, azim=-90, elev=15)
vis_bones = [
    [8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[11,12],[12,13],
    [8,7],[7,0], [0,4],[4,5],[5,6], [0,1],[1,2],[2,3]
]
vis_bones_half = [
    [8,0], [8,14],[8,11], [11,12],[12,13], [14,15],[15,16]
] # half
vis_bones_q = [
    [0,1],[1,2],[1,3], [2,4],[4,5], [3,6],[6,7]
] # q_pose
# vis_bones = [
    # [0,1],[0,2],[0,3]
# ] # torso

# valid_columns = ['chest', 'lu', 'lf', 'ru', 'rf']
valid_columns = ['Chest', 'LeftUpperArm', 'LeftForeArm', 'RightUpperArm', 'RightForeArm']
bones = np.array([
    [0,8], # spine
    [8,11],[8,14], # shoulder
    [11,12],[12,13], # left arm
    [14,15],[15,16] # right arm
])
unity_rot = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1]
], dtype=np.float32)
camera_rot = np.array([
    [-0.91369665,  0.40639689,  0.        ],
    [-0.40639689, -0.91369665,  0.        ],
    [ 0.        ,  0.        ,  1.        ]
], dtype=np.float32)


def read_csv(file_path):
    # reading and parsing data
    data = pd.read_csv(file_path)[valid_columns].to_numpy()
    N, f = data.shape
    res = np.zeros((1,N,f,4))
    for i in range(N):
        for j in range(f):
            res[0][i][j] = np.array([float(val) for val in data[i][j].split('#')])
    return res

def get_camera_rotation():
    rot = np.array([-0.15007018, -0.7552408, 0.62232804, 0.14070565], dtype=np.float32)
    v = Rotation.from_quat(rot).as_matrix() @ np.array([0,0,-1])
    t = normalize(v[:2])
    rad = np.pi*2 - (np.arccos(t[0]) + np.pi/2)
    return get_rmatrix2d(rad)

def get_rmatrix2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def normalize(x):
    return x / np.linalg.norm(x)

def get_angle_xy(u, v):
    theta = np.arccos(u @ v)
    sign = np.arcsin(np.cross(u, v)[2])
    if sign < 0:
        theta = 2 * np.pi - theta
    return theta

def quat_rotation(u, v):
    q = np.empty(4)
    q[:3] = np.cross(u, v)
    q[3] = np.sqrt((u@u) * (v@v)) + u@v
    return normalize(q)

def quat_inv(q):
    q[3] = -q[3]
    return q

def pose_to_quat(pose):
    # get torso vectors
    spine = normalize(pose[8] - pose[0])
    tmp_facing = normalize(
        (pose[11] + pose[14]) / 2 - pose[8]
    )
    # tmp_facing = normalize(
        # (pose[12] - pose[11] + pose[15] - pose[14]) / 2
    # )
    facing = normalize(np.cross(
        normalize(np.cross(spine, tmp_facing)),
        spine
    ))
    # facing_direction = pose[9] - pose[8]
    # if (facing_direction @ facing) < 0:
        # facing = -facing
    
    # get quat in right handed coordinate
        # torso
    target1 = np.array([0,0,1])
    q1 = quat_rotation(spine, target1)
    q1 = Rotation.from_quat(q1)
    
    facing = q1.as_matrix() @ facing
    facing[2] = 0
    facing = normalize(facing)
    target2 = np.array([0,1,0])
    theta = get_angle_xy(facing, target2)
    q2 = Rotation.from_quat(np.array([
        0,
        0,
        np.sin(theta/2),
        np.cos(theta/2)
    ]))
    q = q2 * q1
    quat = [q.inv().as_quat()]
    
        # limbs
    target3 = np.array([0,0,-1])
    for bone in bones[3:]:
        vec = normalize(pose[bone[1]] - pose[bone[0]])
        q_inv = quat_inv(quat_rotation(vec, target3))
        quat.append(q_inv)
    
    # right handed to left handed coordinate
    quat = np.array(quat, dtype=np.float32)
    quat[:,:3] = -quat[:,:3]
    quat = quat[:,[0,2,1,3]]
    return quat

def poses_to_quats(poses):
    assert len(poses.shape) == 3 or len(poses.shape) == 4, \
        'poses_to_quats: invalid poses shape'
    
    # rotate camera view
    poses_copy = poses.copy()
    poses_copy = poses_copy @ np.transpose(camera_rot)
    
    if len(poses_copy.shape) == 3:
        # poses = (frames, joints, 3)
        ret = np.empty((len(poses_copy),5,4))
        for i in range(len(poses_copy)):
            ret[i] = pose_to_quat(poses_copy[i])
        return ret
    else:
        # poses = (sequences, frames, joints, 3)
        k, n, _, _ = poses_copy.shape
        ret = np.empty((k,n,5,4))
        for j in range(k):
            for i in range(n):
                ret[j][i] = pose_to_quat(poses_copy[j][i])
        return ret

def quat_to_vec(quat, direction, length=1):
    if not isinstance(quat, np.ndarray):
        quat = np.array(quat)
    if not isinstance(direction, np.ndarray):
        direction = np.array(direction)
        
    # left handed to right handed coordinates
    quat[:3] = -quat[:3]
    quat = quat[[0,2,1,3]]
    rmat = Rotation.from_quat(quat).as_matrix()
    
    return direction @ np.transpose(rmat)

def quat_to_pose(quat, bone_len, rigid_torso, rigid_hip):
    # left handed to right handed coordinates
    quat[:,:3] = -quat[:,:3]
    quat = quat[:,[0,2,1,3]]
    rmat = []
    for i in range(len(quat)):
        rmat.append(Rotation.from_quat(quat[i]).as_matrix())
    
    # get pose torso
    pose = [[0,0,0]]
    torso = rigid_torso @ np.transpose(rmat[0])
    pose.append(torso[0] * bone_len[0]) # spine (1)
    pose.append(pose[1] + torso[1] * bone_len[1]) # left shoulder (2)
    pose.append(pose[1] + torso[2] * bone_len[2]) # right shoulder (3)
    
    # get pose limbs
    obj = np.array([0,0,-1])
    pose.append(pose[2] + rmat[1] @ obj * bone_len[3]) # left upper arm (4)
    pose.append(pose[4] + rmat[2] @ obj * bone_len[4]) # left lower arm (5)
    pose.append(pose[3] + rmat[3] @ obj * bone_len[5]) # right upper arm (6)
    pose.append(pose[6] + rmat[4] @ obj * bone_len[6]) # right lower arm (7)
    
    # get pose hip
    hip = rigid_hip @ np.transpose(rmat[5])
    pose.append(hip[0] * bone_len[7]) # left hip (8)
    pose.append(hip[1] * bone_len[8]) # right hip (9)
    
    # get legs
    pose.append(pose[8] + rmat[6] @ obj * bone_len[9]) # left thigh (10)
    pose.append(pose[10] + rmat[7] @ obj * bone_len[10]) # left calf (11)
    pose.append(pose[9] + rmat[8] @ obj * bone_len[11]) # right thigh (12)
    pose.append(pose[12] + rmat[9] @ obj * bone_len[12]) # right calf (13)
    
    return np.array(pose)

def quats_to_poses(quats, bone_len, rigid_torso, rigid_hip):
    assert len(quats.shape) == 3 or len(quats.shape) == 4, \
        'quats_to_poses: invalid quats shape'
    
    quats_copy = quats.copy()
    if len(quats_copy.shape) == 3:
        # quats = (frames, joints, 4)
        ret = np.empty((len(quats_copy),14,3))
        for i in range(len(quats_copy)):
            ret[i] = quat_to_pose(
                quats_copy[i], bone_len, rigid_torso, rigid_hip
            )
        return ret
    else:
        # quats = (sequences, frames, joints, 4)
        k, n, _, _ = quats_copy.shape
        ret = np.empty((k,n,14,3))
        for j in range(k):
            for i in range(n):
                ret[j][i] = quat_to_pose(
                    quats_copy[j][i], bone_len, rigid_torso, rigid_hip
                )
        return ret

def plotSke(pts, show=True, is_q=True):
    if is_q:
        vis_bones = vis_bones_q
    else:
        vis_bones = vis_bones_half
    plt3D.clear()
    radius = 1.7
    for i, p in enumerate(vis_bones):
        color = colors[i%len(colors)]
        xs = [pts[p[0]][0], pts[p[1]][0]]
        ys = [pts[p[0]][1], pts[p[1]][1]]
        zs = [pts[p[0]][2], pts[p[1]][2]]
        lims = [[-radius,radius], [-radius,radius], [0,radius]]
        zorder = 3
        plt3D.plot(xs,ys,zs,lims=lims, zdir='z', marker='o', linewidth=3, zorder=zorder, markersize=2, color=color)
    
    if show:
        plt3D.show()
    else:
        img = plt3D.update(require_img=True)
        return img
    return None

def plotVec(vecs, bones=[[0,1],[1,2]], show=True):
    plt3D.clear()
    radius = 1.7
    
    for i, p in enumerate(bones):
        color = colors[i%len(colors)]
        xs = [vecs[p[0]][0], vecs[p[1]][0]]
        ys = [vecs[p[0]][1], vecs[p[1]][1]]
        zs = [vecs[p[0]][2], vecs[p[1]][2]]
        lims = [[-radius,radius], [-radius,radius], [0,radius]]
        zorder = 3
        plt3D.plot(xs,ys,zs,lims=lims, zdir='z', marker='o', linewidth=3, zorder=zorder, markersize=2, color=color)
    
    if show:
        plt3D.show()
    else:
        img = plt3D.update(require_img=True)
        return img
    return None


if __name__ == '__main__':
    pose_vec = read_csv('../data/quaternions/CSV_DATA_20230525_111831.csv')
    
    
    # print(test_front)
    # test_theta = get_angle_xy(front, test_front)
    # test_spine = test_spine / np.linalg.norm(test_spine) * test_theta
    # rotate_rigid_torso(rigid_torso, test_spine)
    pass