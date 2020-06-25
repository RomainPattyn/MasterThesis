import os
import pickle
import sys
import time
from sys import stdout

import cv2
import numpy as np

# Import the pyflow implementation library : path to pyflow folder downloaded at : https://github.com/pathak22/pyflow

LOADING_BAR_CST = 5
PYFLOW_PATH = "/Users/rompat/Documents/MesFichiers/Universite/Master/M2/Memoire/Project/lib"
PATH_TO_PROJECT = os.path.dirname(os.path.abspath(__file__)) + "/example/data/"


class FlowMemory:
    def __init__(self, images, unique_identifier, method="farneback"):
        self.ref_img = images[0]
        self.flow_to_first_file = PATH_TO_PROJECT + "flow_to_first_brox_" + unique_identifier + ".pkl"
        self.flow_consecutives_file = PATH_TO_PROJECT + "flow_consecutives_brox_" + unique_identifier + ".pkl"
        self.height, self.width = self.ref_img.shape
        assert method == "farneback" or method == "brox"
        self.flow_to_first = None
        self.flow_consecutives = None
        if method == 'farneback':
            self.flow_to_first = [
                np.flip(cv2.calcOpticalFlowFarneback(prev=self.ref_img, next=img, flow=None, pyr_scale=0.5,
                                                     levels=2, winsize=7, iterations=3, poly_n=5, poly_sigma=1.1,
                                                     flags=0), axis=2) for img in images[1:]]
            self.flow_consecutives = [np.flip(
                cv2.calcOpticalFlowFarneback(prev=images[i], next=images[i + 1], flow=None, pyr_scale=0.5, levels=2,
                                             winsize=7, iterations=3, poly_n=5, poly_sigma=1.1, flags=0), axis=2)
                for i in range(len(images) - 1)]
        elif method == 'brox':
            if os.path.exists(self.flow_to_first_file) and os.path.exists(self.flow_consecutives_file):
                print("yep")
                # -1 because if we have a sequence of images of length D we will only be able to compute D-1 flows
                self.flow_to_first = pickle.load(open(self.flow_to_first_file, 'rb'))[:len(images) - 1]
                self.flow_consecutives = pickle.load(open(self.flow_consecutives_file, 'rb'))[:len(images) - 1]
            else:
                sys.path.append(os.path.abspath(PYFLOW_PATH + "/pyflow"))
                import pyflow
                alpha = 0.012
                ratio = 0.75
                minWidth = 20
                nOuterFPIterations = 7
                nInnerFPIterations = 1
                nSORIterations = 30
                colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
                pyflow_images = [(img.astype(float) / 255.).reshape((img.shape[0], img.shape[1], 1)) for img in images]
                u_v_img_2_first = []
                u_v_img_2_previous = []
                print("." * ((len(pyflow_images) - 1) // LOADING_BAR_CST))
                s = time.time()
                for i in range(len(pyflow_images) - 1):
                    u_v_img_2_first.append(
                        pyflow.coarse2fine_flow(pyflow_images[0], pyflow_images[i + 1], alpha, ratio, minWidth,
                                                nOuterFPIterations,
                                                nInnerFPIterations, nSORIterations, colType))
                    u_v_img_2_previous.append(
                        pyflow.coarse2fine_flow(pyflow_images[i], pyflow_images[i + 1], alpha, ratio, minWidth,
                                                nOuterFPIterations,
                                                nInnerFPIterations, nSORIterations, colType))
                    if i % LOADING_BAR_CST == 0:
                        stdout.write('.')
                        stdout.flush()
                stdout.write("\tExec t: " + str(time.time() - s) + "\n")
                self.flow_to_first = [np.concatenate((v[..., None], u[..., None]), axis=2) for u, v, _ in
                                      u_v_img_2_first]
                self.flow_consecutives = [np.concatenate((v[..., None], u[..., None]), axis=2) for u, v, _ in
                                          u_v_img_2_previous]
                pickle.dump(self.flow_to_first, open(self.flow_to_first_file, 'wb+'))
                pickle.dump(self.flow_consecutives, open(self.flow_consecutives_file, 'wb+'))

        self.coo = np.flip(np.array(np.meshgrid(range(self.width), range(self.height), indexing='ij')).T, axis=2)
        self.displacement_to_first = [
            np.clip(np.int32(flow + self.coo + 0.5), (0, 0), (self.height - 1, self.width - 1)) for flow in
            self.flow_to_first]
        self.displacement_to_previous = [
            np.clip(np.int32(flow + self.coo + 0.5), (0, 0), (self.height - 1, self.width - 1)) for flow in
            self.flow_consecutives]

        self.consecutive_magnitude, self.consecutive_angle = zip(
            *[cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True) for flow in self.flow_consecutives])
        self.mean_consecutive_distances = self.consecutive_magnitude[0]
        for i, (h, w) in enumerate(self.get_coordinates_after_flow()):
            if i == len(self.consecutive_magnitude) - 1:
                break
            self.mean_consecutive_distances += self.consecutive_magnitude[i + 1][h, w]
        self.mean_consecutive_distances /= len(self.consecutive_magnitude)

        self.to_first_magnitude, self.to_first_angle = zip(
            *[cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True) for flow in self.flow_to_first])
        self.mean_to_first_distances = self.to_first_magnitude[0]
        for i, (h, w) in enumerate(self.get_coordinates_after_flow()):
            if i == len(self.to_first_magnitude) - 1:
                break
            self.mean_to_first_distances += self.to_first_magnitude[i + 1][h, w]
        self.mean_to_first_distances /= len(self.to_first_magnitude)

    def get_coordinates_after_flow(self, coordinates=None):
        """
        :param shift: is a tuple that is added to the cooordinates to find the right entries and that is substracted
        from to output so that it matches the domain from where this function was called
        :param coordinates: an array containing the coordinates wanted
        :param bounds: (h_min, h_max, w_min, w_max) which are the coordinates of the sub img wanted
        :return: yield the displacements tuple. The first displacement is identity
        """
        if coordinates is not None:
            h, w = coordinates
            for index in range(len(self.flow_to_first)):
                yield self.displacement_to_first[index][h, w].T
        else:
            for i in range(len(self.flow_to_first)):
                yield self.get_coordinates_after_flow_index(i)

    def get_coordinates_after_flow_index(self, index):
        result = self.displacement_to_first[index]
        return result[..., 0], result[..., 1]
