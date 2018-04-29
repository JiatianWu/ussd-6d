import cv2
from docopt import docopt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from ssd.ssd_utils import load_frozen_graph, NMSUtility, process_detection_output
from rendering.utils import precompute_projections, build_6D_poses, verify_6D_poses
from rendering.utils import draw_detections_2D, draw_detections_3D
from utils.sixd import *

def cal_bb_gt_32(path):
    model = load_model_info('models/models_info.yml')
    for seq in range(1,16):
        path_K = os.path.join(path, '{:02d}/info.yml'.format(seq))
        path_RT = os.path.join(path, '{:02d}/gt.yml'.format(seq))
        path_bb = os.path.join(path, '{:02d}/bb.pkl'.format(seq))
        info_K = load_info(path_K)
        info_RT = load_gt(path_RT)

        bb_gt_32 = {}
        for eid in range(0,1313):
            K = info_K[eid]['cam_K']
            R = info_RT[eid][0]['cam_R_m2c']
            T = info_RT[eid][0]['cam_t_m2c']

            bb = model['{:02d}'.format(seq)].bb
            bb_32d_all = []
            for bid in range(0,8):
                bbv = np.array(bb[bid]).reshape(3,1)
                bb_3d = np.matmul(R, bbv)
                bb_3d += T
                bb_32d = np.matmul(K, bb_3d)
                bb_32d = bb_32d / bb_32d[2]
                bb_32d_all.append(bb_32d[0:2])

            bb_gt_32[str(eid)] = bb_32d_all

        with open(path_bb, 'w') as f:
            pickle.dump(bb_gt_32, f,  pickle.HIGHEST_PROTOCOL)

def check_bb_gt_32(path):
    for seq in range(1,16):
        path_bb = os.path.join(path, '{:02d}/bb.pkl'.format(seq))
        bb_object = open(path_bb, 'r')
        bb = pickle.load(bb_object)
        print bb['0']

def main():
    check_bb_gt_32('train/')

if __name__== "__main__":
  main()