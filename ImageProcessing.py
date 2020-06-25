import functools
import importlib
import math
import operator
import os
import pickle as pkl
import time
from sys import stdout

import FlowMemory
import Grid
import Tools
import VideoViewer
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import iqr
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, erosion, square

sns.set_palette("bright")
sns.set_style("darkgrid")

importlib.reload(VideoViewer)
importlib.reload(Grid)
importlib.reload(Tools)

LOADING_BAR_CST = 5
# PATH_TO_SAVE_IMAGES = "/Users/rompat/Documents/MesFichiers/Universite/Master/M2/Memoire//Rapport/Images/"
PATH_TO_SAVE_IMAGES = os.path.dirname(os.path.abspath(__file__)) + "/images/"
PATH_TO_VARIABLES = os.path.dirname(os.path.abspath(__file__)) + "/vars/variables"
FONT = {'family': 'serif',
        'weight': 'normal',
        'size': 20,
        }


def get_number_image_sets(file):
    if file == "1Plan1Pos":
        return 1
    if file == "1Plan3Pos":
        return 3
    if file == "2Plans":
        return 2
    if file == "2PlansBis":
        return 2


def save_navigators(MR_IMG, navigators, mask_labels, save_folder_path):
    opacity = 0.6
    ref_img = cv2.cvtColor(MR_IMG, cv2.COLOR_GRAY2RGB)
    img = ref_img.copy()
    h, w = np.argwhere(mask_labels > 0).T
    img[h, w] = np.array([173, 255, 47])
    img = cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0)
    for p0, p1 in navigators:
        cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), (250, 0, 0), 1)
    plt.imsave(save_folder_path, img, dpi=1000)


def Generate_Image_Folder(folder_path):
    if not os.path.exists(folder_path + "/SendToRomain"):
        os.makedirs(folder_path + "/SendToRomain")
    for dir_tuple in os.walk(folder_path):
        for dir in dir_tuple[1]:
            for sub_dir_tuple in os.walk(folder_path + "/" + dir):
                for sub_sub_dir in sub_dir_tuple[1]:
                    num = get_number_image_sets(sub_sub_dir)
                    for mr in range(num, num * 2):
                        ct = mr + num
                        file_path = folder_path + "/" + dir + "/" + sub_sub_dir + "/" + sub_sub_dir + ".p"
                        MR, CT = GET_IMAGES(data_path=file_path, mr=mr, ct=ct)
                        t_s = time.time()
                        navigators, mask_labels = GET_NAVIGATORS(MR[:30], CT, return_regions_mask=True)
                        print("Execution time: " + str(time.time() - t_s) + "\tLength MR sequence" + str(len(MR)))
                        save_navigators(MR[0], navigators, mask_labels,
                                        folder_path + "/SendToRomain/" + dir + "_" + sub_sub_dir + "_" + str(
                                            mr) + ".png")
        break

def GET_IMAGES(data_path, mr=2, ct=4):
    if data_path.endswith(".p"):
        max_bytes = 2 ** 31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(data_path)
        with open(data_path, 'rb') as file:
            for _ in range(0, input_size, max_bytes):
                bytes_in += file.read(max_bytes)
        dataDictionary = pkl.loads(bytes_in)
        print("Data dictionary loaded with success")
        return dataDictionary['dynamic2DImageList'][mr]['imageArrayList'], dataDictionary['dynamic2DImageList'][ct][
            'imageArrayList']
    else:
        print("Data format not correct, please select a .p file")


def GET_NAVIGATORS(MRI_images, CT_images, identifier="None", return_regions_mask=False, show_figures=False, save_images=False,
                   flow_method='farneback', number_groups=None):
    process = ImageProcessing(MRI_images, CT_images, identifier=identifier, show_figures=show_figures, print_details=False,
                              flow_method=flow_method, save_images=save_images)
    X = process.get_navigators(return_regions_mask=return_regions_mask, number_groups=number_groups)
    return X


def GET_REGIONS(MRI_images, CT_images, identifier="None", show_figures=False, save_images=False, flow_method='farneback',
                number_groups=None):
    process = ImageProcessing(MRI_images, CT_images, identifier=identifier, show_figures=show_figures, print_details=False,
                              flow_method=flow_method, save_images=save_images)
    X = process.region_selection_phase(number_groups)
    return X


def SHOW_NAVIGATORS(REF_IMG, navigators_coordinates, mask_labels=None):
    opacity = 0.6
    ref_img = cv2.cvtColor(REF_IMG, cv2.COLOR_GRAY2RGB)
    img = ref_img.copy()
    if mask_labels is not None:
        h, w = np.argwhere(mask_labels > 0).T
        img[h, w] = np.array([173, 255, 47])
        img = cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0)
    plt.figure(figsize=(10, 20))
    plt.axis("off")
    for p0, p1 in navigators_coordinates:
        cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), (250, 0, 0), 1)
    # plt.imsave(PATH_TO_SAVE_IMAGES + "example_step_5.png", img, dpi=600)
    plt.imshow(img)

class ImageProcessing:
    """
        The patient index is only used when the Brox's algorithm is used to estimate de displacement field.
        That idx is used to save the displacement fields locally.
    """

    def __init__(self, mri_images, ct_images, identifier="None", show_figures=False, print_details=False, num_frames=None,
                 flow_method='farneback', save_images=False):
        assert(flow_method == 'farneback' or (flow_method == 'brox' and identifier != "None"))
        self.show_figures = show_figures
        self.print_details = print_details
        self.save_images = save_images
        self.external_variables = self.load_external_variables()
        self.mri_images = mri_images
        self.image_height, self.image_width = self.mri_images[0].shape
        self.ct_images = ct_images
        dim = self.mri_images[0].shape
        self.ct_images = [cv2.resize(img, (dim[1], dim[0])) for img in self.ct_images]
        if num_frames is None:
            num_frames = len(self.mri_images)
        assert num_frames <= len(self.mri_images)
        self.Images = [cv2.GaussianBlur(img, (3, 3), 0) for img in self.mri_images[:num_frames]]
        self.mean_gradient_angle_MRI = None
        self.Flow_Memory = FlowMemory.FlowMemory(self.Images, identifier, method=flow_method)
        self.VideoViewer = VideoViewer.VideoViewer(self.Images, self.Flow_Memory)
        self.number_changements = None
        self.VideoViewer.add_layer('flow', [4, 1], color='white')

        self.print_if_wanted("Generating CT seq closest to MRI seq according to Mutual Information.")
        s = time.time()
        self.ct_mi_mri_seq, self.ct_mi_mri_seq_index = self.get_closest_mi_ct_seq()
        gradient_h = [cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5) for img in self.ct_images[0:10]]
        gradient_w = [cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5) for img in self.ct_images[0:10]]
        self.ct_loop_gradients_magnitudes, _ = zip(
            *[cv2.cartToPolar(g_h, g_w, angleInDegrees=True) for g_h, g_w in zip(gradient_h, gradient_w)])
        self.print_if_wanted("Elapsed time : " + str(time.time() - s) + "\n")
        self.VideoViewer.add_layer('ct_correspondance', self.ct_mi_mri_seq)

    def print_if_wanted(self, str):
        if self.print_details:
            stdout.write(str)
            stdout.flush()

    def print_loading_bar(self, img_number):
        if self.print_details and img_number % LOADING_BAR_CST == 0:
            stdout.write('.')
            stdout.flush()

    def load_external_variables(self):
        res = {}
        with open(PATH_TO_VARIABLES) as f:
            for line in f:
                (key, val) = line.split()
                if key == "min_average_displacement":
                    res[key] = float(val)
                else:
                    res[key] = int(val)
        return res

    def get_closest_mi_ct_seq(self):
        """
        :return: A tuple where the first element is a sequence of CT images that are the closest (using the mutual
        information criterion) to the MRI sequence.
        The second element of the tuple is a list of the same length as the MRI sequence that contains ints in [0,9]
        where the i-th int 'j' of that list is saying the the closest CT image to the i-th MRI image is the j-th of the
        10 CT images
        """
        ct_seq = []
        ct_index_seq = []
        for i, img_mri in enumerate(self.Images):
            best_ct = None
            best_mi = 0
            best_index = 0
            for j, img_ct in enumerate(self.ct_images[0:10]):
                hist_2d, _, _ = np.histogram2d(img_mri.ravel(), img_ct.ravel(), bins=20)
                mi = Tools.mutual_information(hist_2d)
                if best_mi < mi:
                    best_mi = mi
                    best_ct = img_ct
                    best_index = j
            ct_seq.append(best_ct)
            ct_index_seq.append(best_index)
        return ct_seq, ct_index_seq

    def get_navigators(self, number_groups=None, show_region_properties=False, return_regions_mask=False):
        """
        :param number_groups:
        :param show_region_properties: (boolean) True if we want to show the regions, False otherwise
        :return: a list of elements composed of two coordinates representing the starting and end point of the
        navigators in the form : [[(w1, h1),(w2, h2)], ...]
        """
        # Selection of the interesting regions
        s = time.time()
        mask_labels, points = self.region_selection_phase(number_groups)
        e = round(time.time() - s, 2)
        regions = regionprops(mask_labels)
        self.print_if_wanted("Region Selection Phase : \t\t Elapsed time :" + str(e) + "\n")
        self.print_if_wanted("\t- " + str(len(regions)) + " regions selected")

        # Show the region properties like the ellipses axes
        if self.show_figures and show_region_properties:
            self.show_regions_properties(regions)

        # Positioning the navigators on the regions
        s = time.time()
        navigators_coordinates = self.navigators_positioning_phase_2(mask_labels)
        e = round(time.time() - s, 2)

        self.print_if_wanted("Navigators Positioning Phase : \t\t Elapsed time :" + str(e) + "\n")
        if return_regions_mask:
            return navigators_coordinates, mask_labels
        return navigators_coordinates

    # -----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Regions Selection Phase --------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def region_selection_phase(self, number_groups):
        """
        :return:    - a mask that contains a unique label for pixels belonging to the same edge
                    - a mask containing the mean angle differences per interest point
                    - a list of coordinates containing the interesting points
        """

        # Computing the gradients values and angles for every MRI image of the sequence
        ksize = self.external_variables["mri_sobel_ksize"]
        gradient_h = [cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize) for img in self.Images]
        gradient_w = [cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize) for img in self.Images]
        grad_mag, grad_angle = zip(
            *[cv2.cartToPolar(g_h, g_w, angleInDegrees=True) for g_h, g_w in zip(gradient_h, gradient_w)])

        # STEP 1 : Locating and adding the edges points to the Video_Viewer
        h, w = self.step_1_moving_edges_coordinates(grad_mag, grad_angle)
        self.show_edges(h, w, color='light_green')

        # STEP 2 : Angle Discrepancy Filter
        angle_errors = [np.minimum((360 - f_a + g_a) % 180, (360 - g_a + f_a) % 180) for f_a, g_a in
                        zip(self.Flow_Memory.consecutive_angle, grad_angle)]
        error_per_coo = self.step_2_angle_discrepancy_matrix(angle_errors, h, w)
        mask_step_2 = self.step_2_filter_angle_discrepancy(h, w, error_per_coo, 'light_green')

        # STEP 3 : CT Filtering
        mask_ct_edge_ok = self.step_3_filter_non_ct_edges(mask_step_2, 'light_green')

        # STEP 4 : Blob Size Thresholding
        mask_labels, mask_size_ok, points = self.step_4_filter_blob_size(mask_ct_edge_ok, number_groups, 'light_green')

        maks_grad_ang_1 = np.mod(np.where(mask_labels > 0, self.mean_gradient_angle_MRI, 0), 180)
        binary_mask = np.where(mask_labels > 0, 1, 0).astype(float)
        h2, w2 = np.argwhere(mask_labels > 0).T
        kernel = np.ones((5, 5), np.float32)
        active_pixels = cv2.filter2D(binary_mask, -1, kernel)
        grad_sum = cv2.filter2D(maks_grad_ang_1, -1, kernel)
        smooth_grad_values = grad_sum[h2, w2] / active_pixels[h2, w2]
        maks_grad_ang_2 = np.copy(maks_grad_ang_1)
        maks_grad_ang_2[h2, w2] = smooth_grad_values
        if self.show_figures:
            plt.figure(figsize=(16, 10), dpi=80)
            plt.subplot(151)
            plt.imshow(mask_step_2, cmap=plt.cm.gray)
            plt.title(r"$\mu_{\alpha_e}$" + " before ct matching")
            plt.axis("off")
            plt.subplot(152)
            plt.imshow(mask_ct_edge_ok, cmap=plt.cm.gray)
            plt.title(r"$\mu_{\alpha_e}$" + " after ct matching")
            plt.axis("off")
            plt.subplot(153)
            plt.imshow(mask_size_ok, cmap=plt.cm.gray)
            plt.title(r"$\mu_{\alpha_e}$" + " after ct matching and closing")
            plt.axis("off")
            plt.subplot(154)
            # plt.imshow(mask_labels, cmap="tab20")
            plt.imshow(maks_grad_ang_1, cmap=plt.cm.inferno)
            plt.title("Labeled image")
            plt.axis("off")
            plt.subplot(155)
            plt.imshow(maks_grad_ang_2, cmap=plt.cm.inferno)
            plt.title("Gradient Angle")
            plt.axis("off")
            plt.show()

        if self.save_images:
            opacity = 0.6
            ref_img = cv2.cvtColor(self.Images[0], cv2.COLOR_GRAY2RGB)
            my_dpi = 192
            plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
            plt.imshow(ref_img)
            plt.axis("off")
            plt.savefig(PATH_TO_SAVE_IMAGES + "example_step_0.png", dpi=my_dpi*10, bbox_inches="tight")
            plt.close()

            img = ref_img.copy()
            img[h, w] = np.array([173, 255, 47])
            plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
            plt.imshow(cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0))
            plt.axis("off")
            plt.savefig(PATH_TO_SAVE_IMAGES + "example_step_1.png", dpi=my_dpi, bbox_inches="tight")
            plt.close()

            h2, w2 = np.argwhere(mask_step_2 > 0).T
            img = ref_img.copy()
            img[h2, w2] = np.array([173, 255, 47])
            plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
            plt.imshow(cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0))
            plt.axis("off")
            plt.savefig(PATH_TO_SAVE_IMAGES + "example_step_2.png", dpi=my_dpi, bbox_inches="tight")
            plt.close()

            h3, w3 = np.argwhere(mask_ct_edge_ok > 0).T
            img = ref_img.copy()
            img[h3, w3] = np.array([173, 255, 47])
            plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
            plt.imshow(cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0))
            plt.axis("off")
            plt.savefig(PATH_TO_SAVE_IMAGES + "example_step_3.png", dpi=my_dpi, bbox_inches="tight")
            plt.close()

            h4, w4 = np.argwhere(mask_size_ok > 0).T
            img = ref_img.copy()
            img[h4, w4] = np.array([173, 255, 47])
            plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
            plt.imshow(cv2.addWeighted(img, opacity, ref_img, 1 - opacity, 0))
            plt.axis("off")
            plt.savefig(PATH_TO_SAVE_IMAGES + "example_step_4.png", dpi=my_dpi*10, bbox_inches="tight")
            plt.close()

        # angle_errors_counts = [(a_e <= 90).astype(int) for a_e in angle_errors]
        # self.number_changements = [np.sum(np.absolute(angle_errors_1_counts[i] - angle_errors_1_counts[i + 1])) for i in range(len(angle_errors_1_counts) - 1)]

        return mask_labels, points

    def step_1_moving_edges_coordinates(self, gradients_magnitudes, gradients_angles,
                                        show_edges_movement_thresholds=True):
        """
        :param gradients_magnitudes: (list(arrays)) List of arrays of the size of the MRI images containing the
        gradients magnitudes of each pixel of each frame of the MRI sequence
        :param gradients_angles: (list(arrays)) List of arrays of the size of the MRI images containing the
        gradients angles of each pixel of each frame of the MRI sequence
        :param show_edges_movement_thresholds: (boolean) : True if we want to show the result of different thresholds
        on the mean consecutive movement
        :return: (1D-array, 1D-array) Corresponding to respectively the height and the width values of the edge points
        """

        # Computing the mean gradient magnitude and angle in each point

        kernel = np.ones((3, 3), np.float32) / 9
        cumulative_gradients_mag = gradients_magnitudes[0].copy()
        cumulative_gradients_angles = gradients_angles[0].copy()
        cumulative_gradients_mag_ct = self.ct_loop_gradients_magnitudes[self.ct_mi_mri_seq_index[0]].copy()

        for idx, (h, w) in enumerate(self.Flow_Memory.get_coordinates_after_flow()):
            cumulative_gradients_mag += gradients_magnitudes[idx + 1][h, w]
            cumulative_gradients_angles += gradients_angles[idx + 1][h, w]
            cumulative_gradients_mag_ct += self.ct_loop_gradients_magnitudes[self.ct_mi_mri_seq_index[idx + 1]][h, w]
        cumulative_gradients_mag /= len(gradients_magnitudes)
        cumulative_gradients_angles /= len(gradients_angles)
        cumulative_gradients_mag_ct /= len(self.ct_mi_mri_seq_index)
        self.mean_gradient_angle_MRI = cumulative_gradients_angles
        """
        for i in range(10):
            plt.imsave(PATH_TO_SAVE_IMAGES + "grad_mag_" + str(i) + ".png", gradients_magnitudes[i], dpi=600, cmap=plt.cm.gray)
            plt.imsave(PATH_TO_SAVE_IMAGES + "grad_orig_" + str(i) + ".png", self.Images[i], dpi=600, cmap=plt.cm.gray)
        plt.imsave(PATH_TO_SAVE_IMAGES + "grad_disp_mask.png", np.where(cumulative_gradients_mag < 30, 1, 0), dpi=600, cmap=plt.cm.gray)
        """
        # Showing the parts that are kept and rejected with different thresholds on the mean movement amplitude
        if self.show_figures and show_edges_movement_thresholds:
            thresholds = [0.6, 0.4, 0.2]
            number_plots = len(thresholds) * 2 - 1
            figure = plt.figure(figsize=(16, 10), dpi=80)
            figure.add_subplot(1, number_plots, 1)
            plt.title("Mean gradient magnitudes")
            plt.imshow(cumulative_gradients_mag, cmap=plt.cm.inferno)
            plt.axis("off")
            figure.add_subplot(1, number_plots, 2)
            plt.title("Mean displacement magnitudess")
            plt.imshow(self.Flow_Memory.mean_consecutive_distances, cmap=plt.cm.inferno)
            plt.axis("off")
            for i, thresh in enumerate(thresholds):
                h, w = zip(*np.argwhere(self.Flow_Memory.mean_consecutive_distances < thresh))
                excluded = np.zeros((self.image_height, self.image_width))
                excluded[h, w] = 1
                figure.add_subplot(1, number_plots, i + 2 + 1)
                plt.title("Threshold : " + str(thresh))
                plt.imshow(excluded, cmap=plt.cm.gray)
                if self.save_images:
                    plt.imsave(PATH_TO_SAVE_IMAGES + "threshold_movement_magnitude_0" + str(
                        int(10 * thresh)) + ".png", excluded, dpi=600, cmap=plt.cm.gray)
                plt.axis("off")
            if self.save_images:
                plt.imsave(PATH_TO_SAVE_IMAGES + "matrix_gradient_magnitude.png",
                           cumulative_gradients_mag, dpi=600, cmap=plt.cm.inferno)
                plt.imsave(PATH_TO_SAVE_IMAGES + "matrix_movement_magnitude.png",
                           self.Flow_Memory.mean_consecutive_distances, dpi=600, cmap=plt.cm.inferno)
            plt.show()

        # Putting zeros in the mean gradients on areas that aren't moving enough so that we can use np.argwhere( > 0)
        min_average_displacement = self.external_variables["min_average_displacement"]
        cumulative_gradients = np.where(self.Flow_Memory.mean_consecutive_distances < min_average_displacement, 0,
                                        cumulative_gradients_mag)

        # Identification of the coordinates that are on edges
        # We bring it back in a known interval so that we can make the size of the sobel filter vary
        interval = 10
        threshold = self.external_variables["edge_gradient_threshold_0_10"]
        data = self.bring_in_interval(cumulative_gradients, 0, interval, IQR=97, show_histograms=False)
        data = np.where(data > threshold, 1, 0)
        if self.save_images:
            plt.imsave(PATH_TO_SAVE_IMAGES + "threshold_gradient_magnitude_0.png", data, dpi=600,
                       cmap=plt.cm.gray)
        min_edge_size = self.external_variables["minimum_edge_size"]
        label_image = label(data)

        # We discard pixels that aren't on big enough edges
        for region in regionprops(label_image):
            if region.area < min_edge_size:
                hh, ww = zip(*region.coords)
                data[hh, ww] = 0
        h, w = np.argwhere(data == 1).T
        return h, w

    def step_2_angle_discrepancy_matrix(self, angle_errors, h, w, show_distribution=False):
        errors = angle_errors[0][h, w]
        for img_number, (h_new, w_new) in enumerate(self.Flow_Memory.get_coordinates_after_flow(coordinates=(h, w))):
            self.print_loading_bar(img_number)
            if img_number == len(angle_errors) - 1:
                break
            errors += angle_errors[img_number + 1][h_new, w_new]
        self.print_if_wanted('\n')
        errors /= len(angle_errors)
        if self.show_figures and show_distribution:
            plt.figure(figsize=(16, 10), dpi=80)
            sns.kdeplot(errors, shade=True, label="Mean angle distribution", alpha=.4)
            sns.kdeplot(np.array(angle_errors).flatten(), shade=True, label="Angle Distribution", alpha=.4)
            plt.title('Mean angle vs angle distribution', fontsize=22)
            plt.legend()
            plt.show()
        return errors

    def step_2_filter_angle_discrepancy(self, h, w, error_per_coo, color):
        # Computing a mask with the same size as the original image with zeros at non-interesting points and the value
        # of the angle error at interesting points
        mask_angle_ok = np.zeros((self.image_height, self.image_width))
        max_angle = self.external_variables["max_angle"]
        coo = np.argwhere(error_per_coo < max_angle)
        mask_angle_ok[h[coo], w[coo]] = error_per_coo[coo]
        if self.save_images:
            mask = np.ones((self.image_height, self.image_width))
            mask[h[coo], w[coo]] = 0
            plt.imsave(PATH_TO_SAVE_IMAGES + "binary_mask_step2.png", mask, dpi=600, cmap=plt.cm.gray)
        # Adding the first layer (no filtering yet) of interesting points to the Video_Viewer
        points = np.array([(hh, ww) for hh, ww in np.argwhere(mask_angle_ok > 0)])
        self.VideoViewer.add_layer('following_points', points, opacity=1, color=color)
        return mask_angle_ok

    def show_edges(self, h, w, color):
        mask = np.zeros((self.image_height, self.image_width))
        mask[h, w] = 1.
        points = np.array([(hh, ww) for hh, ww in np.argwhere(mask == 1)])
        self.VideoViewer.add_layer('following_points', points, opacity=1, color=color)

    def step_3_filter_non_ct_edges(self, mask, color, show_densties=False):
        coo = np.argwhere(mask > 0)
        h, w = coo[..., 0], coo[..., 1]
        ct_gradients = self.ct_loop_gradients_magnitudes[self.ct_mi_mri_seq_index[0]][h, w]
        for img_number, (h_new, w_new) in enumerate(self.Flow_Memory.get_coordinates_after_flow(coordinates=(h, w))):
            if img_number == len(self.ct_mi_mri_seq_index) - 1:
                break
            ct_gradients = np.vstack((ct_gradients,
                                      self.ct_loop_gradients_magnitudes[self.ct_mi_mri_seq_index[img_number + 1]][
                                          h_new, w_new]))
        ct_gradients = np.transpose(ct_gradients)
        ct_gradients = np.flip(np.sort(ct_gradients), axis=1)

        if self.show_figures and show_densties:
            plt.figure(figsize=(16, 10), dpi=80)
            y = np.tile(np.arange(ct_gradients.shape[1]), (ct_gradients.shape[0], 1))
            plt.plot(y.T, ct_gradients.T)
            plt.show()
            plt.figure(figsize=(16, 10), dpi=80)
            interval = 60
            for i in range(ct_gradients.shape[1] // interval, 0, -1):
                mean_per_point = np.mean(ct_gradients[:, :i * interval], axis=1)
                percentage = 100 * i * interval / ct_gradients.shape[1]
                sns.kdeplot(mean_per_point, shade=True, label=str(round(percentage, 1)) + " % of the highest gradients",
                            alpha=.4)
            sns.kdeplot(ct_gradients[:, 1], shade=True, label="Highest gradient", alpha=.4)
            # plt.title('Mean Gradient Magnitude Distribution Evolution', fontsize=22)
            plt.legend(prop=FONT)
            plt.xlabel("Mean Gradient Magnitude", fontdict=FONT)
            plt.ylabel("Density", fontdict=FONT)
            if self.save_images:
                plt.savefig(PATH_TO_SAVE_IMAGES + "ct_grad_distribution.png", dpi=600,
                            bbox_inches="tight")
            plt.show()
        ct_threshold = self.external_variables["ct_threshold"]
        nbr_sample_for_mean_ct_gradient = self.external_variables["nbr_sample_for_mean_ct_gradient"]
        shift_for_mean_ct_gradient = self.external_variables["shift_for_mean_ct_gradient"]

        mean_per_point = np.mean(
            ct_gradients[:, shift_for_mean_ct_gradient:shift_for_mean_ct_gradient + nbr_sample_for_mean_ct_gradient],
            axis=1)
        index = np.argwhere(mean_per_point < ct_threshold)
        new_mask = mask.copy()
        new_mask[h[index], w[index]] = 0
        points = np.array([(hh, ww) for hh, ww in np.argwhere(new_mask > 0)])
        self.VideoViewer.add_layer('following_points', points, opacity=1, color=color)
        return new_mask

    def step_4_filter_blob_size(self, mask_ct_edge_ok, number_groups, color):

        # Proceding to a closing of size 4 (which is considered to be big), it will regroup groups that are closer than
        # 4 pixels of one to each other. Then we threshold the groups on their sizes
        square_size_big = self.external_variables["closing_big"]

        if self.save_images:
            before = np.where(mask_ct_edge_ok > 0, 0, 1)
            mask_dilate_big = dilation(np.where(mask_ct_edge_ok > 0, 1, 0), square(4))
            after_dilation = np.where(mask_dilate_big > 0, 0, 1)
            after_erosion = np.where(erosion(mask_dilate_big, square(4)) > 0, 0, 1)
            plt.imsave(PATH_TO_SAVE_IMAGES + "before_closing.png", before, dpi=600, cmap=plt.cm.gray)
            plt.imsave(PATH_TO_SAVE_IMAGES + "after_dilation.png", after_dilation, dpi=600,
                       cmap=plt.cm.gray)
            plt.imsave(PATH_TO_SAVE_IMAGES + "after_erosion.png", after_erosion, dpi=600,
                       cmap=plt.cm.gray)

        mask_closing_big = closing(np.where(mask_ct_edge_ok > 0, 1, 0), square(square_size_big))
        label_image_big = label(mask_closing_big)
        minimum_blob_size_big = self.external_variables["minimum_blob_size_closing_big"]

        mask_after_closing = mask_ct_edge_ok.copy()
        for region in regionprops(label_image_big):
            if region.area < minimum_blob_size_big:
                h, w = zip(*region.coords)
                mask_after_closing[h, w] = 0

        final_labels = label(np.where(mask_after_closing > 0, 1, 0))
        group_sizes = [(region, region.area) for region in regionprops(final_labels)]
        if number_groups is None:
            number_groups = self.external_variables["number_groups"]
        res = sorted(group_sizes, key=lambda i: i[1])
        for region, area in res[:-number_groups]:
            h, w = zip(*region.coords)
            mask_after_closing[h, w] = 0
        """
        # Second closing threshold with a smaller closing limit of 2
        square_size_small = self.external_variables["closing_small"]
        mask_closing_small = closing(np.where(mask_after_closing > 0, 1, 0), square(square_size_small))
        label_image_small = label(mask_closing_small)
        minimum_blob_size_small = self.external_variables["minimum_blob_size_closing_small"]

        for region in regionprops(label_image_small):
            if region.area < minimum_blob_size_small:
                h, w = zip(*region.coords)
                mask_after_closing[h, w] = 0
        """
        labeled_mask = np.where(mask_after_closing > 0, label_image_big, 0)

        points = np.array([(hh, ww) for hh, ww in np.argwhere(mask_after_closing > 0)])
        self.VideoViewer.add_layer('following_points', points, opacity=1, color=color)

        return labeled_mask, mask_after_closing, points

    def show_regions_properties(self, regions):
        plt.figure(figsize=(10, 20))
        plt.imshow(self.Images[0], cmap=plt.cm.gray)
        for props in regions:
            for h, w in props.coords:
                plt.plot(w, h, '.g', markersize=1, alpha=0.9)
            h0, w0 = props.centroid
            orientation = props.orientation
            w1 = w0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            h1 = h0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            w2 = w0 - math.sin(orientation) * 0.5 * props.major_axis_length
            h2 = h0 - math.cos(orientation) * 0.5 * props.major_axis_length

            plt.plot((w0, w1), (h0, h1), '-r', linewidth=1)
            plt.plot((w0, w2), (h0, h2), '-r', linewidth=1)
            plt.plot(w0, h0, '.g', markersize=2)

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            plt.plot(bx, by, '-b', linewidth=1)
        plt.axis("off")
        if self.save_images:
            pass
        plt.savefig(PATH_TO_SAVE_IMAGES + "group_properties.png", dpi=600,
                    bbox_inches="tight")
        plt.show()

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ Navigators Positioning Phase -----------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def navigators_positioning_phase_22(self, mask_labels):
        res = []
        navigator_length = 10
        for region in regionprops(mask_labels):
            h0, w0 = region.centroid
            orientation = region.orientation
            w1 = w0 + math.cos(orientation) * 0.5 * navigator_length
            h1 = h0 - math.sin(orientation) * 0.5 * navigator_length
            w2 = w0 - math.cos(orientation) * 0.5 * navigator_length
            h2 = h0 + math.sin(orientation) * 0.5 * navigator_length
            res.append([(np.int32(w1 + 0.5), np.int32(h1 + 0.5)), (np.int32(w2 + 0.5), np.int32(h2 + 0.5))])
        self.VideoViewer.add_layer('navigators', res, color='yellow')
        print(res)
        return res

    def navigators_positioning_phase_2(self, mask_labels):
        navigator_length = 15
        H, W = zip(*[region.centroid for region in regionprops(mask_labels)])
        H = [int(hh+0.5) for hh in H]
        W = [int(ww+0.5) for ww in W]
        all_h = [H]
        all_w = [W]
        nbr_pts = len(H)
        for hh, ww in self.Flow_Memory.get_coordinates_after_flow(coordinates=(H, W)):
            all_h.append(hh)
            all_w.append(ww)

        all_h = np.array(all_h).T
        all_w = np.array(all_w).T
        mean_h = np.mean(all_h, axis=1).reshape((nbr_pts, 1))
        mean_w = np.mean(all_w, axis=1).reshape((nbr_pts, 1))

        orientations = np.array([region.orientation for region in regionprops(mask_labels)])

        v_h = np.cos(orientations).reshape((nbr_pts, 1))
        v_w = np.sin(orientations).reshape((nbr_pts, 1))

        # Projecting the points on the line that intersects the mean coordinate and that has an angle corresponding
        # to the gradient direction
        dot_prod = (all_h - mean_h) * v_h + (all_w - mean_w) * v_w
        pp_h = dot_prod * v_h
        pp_w = dot_prod * v_w

        # Rotating the data so that we don't need to compute the distances but can simply look at the maximum and
        # minimums taken by a point during the sequence
        cos_rot = np.cos(-orientations).reshape((nbr_pts, 1))
        sin_rot = np.sin(-orientations).reshape((nbr_pts, 1))
        rot_h = pp_h * cos_rot - pp_w * sin_rot

        # Computing the mean extrema of each interest point
        nbr_extrema = 3
        max_extrema_mean = np.mean(np.partition(rot_h, -nbr_extrema, axis=1)[:, -nbr_extrema:], axis=1)
        min_extrema_mean = np.mean(-np.partition(-rot_h, -nbr_extrema, axis=1)[:, -nbr_extrema:], axis=1)
        distances = max_extrema_mean - min_extrema_mean

        """
        factor = 3
        res = [[(int(w0 + math.cos(orientation) * 0.5 * dist * factor + 0.5),
                 int(h0 - math.sin(orientation) * 0.5 * dist * factor + 0.5)),
                (int(w0 - math.cos(orientation) * 0.5 * dist * factor + 0.5),
                 int(h0 + math.sin(orientation) * 0.5 * dist * factor + 0.5))]
               for orientation, h0, w0, dist in zip(orientations, mean_h, mean_w, distances)]
        """

        res = [[(int(w0 + math.cos(orientation) * 0.5 * navigator_length + 0.5),
                 int(h0 - math.sin(orientation) * 0.5 * navigator_length + 0.5)),
                (int(w0 - math.cos(orientation) * 0.5 * navigator_length + 0.5),
                 int(h0 + math.sin(orientation) * 0.5 * navigator_length + 0.5))]
               for orientation, h0, w0 in zip(orientations, mean_h, mean_w)]

        self.VideoViewer.add_layer('navigators', res, color='yellow')
        print(res)
        return res

    def navigators_positioning_phase(self, mask_labels, points, show_projected_distance=False):
        assert (self.mean_gradient_angle_MRI is not None)
        h, w = points.T
        coo_list = [(a, b) for a, b in points.tolist()]
        coo_2_idx = dict(zip(coo_list, list(range(len(points)))))
        nbr_pts = h.size
        all_h = [h]
        all_w = [w]
        for hh, ww in self.Flow_Memory.get_coordinates_after_flow(coordinates=(h, w)):
            all_h.append(hh)
            all_w.append(ww)

        all_h = np.array(all_h).T
        all_w = np.array(all_w).T

        # Recenter the data to the mean coordinate taken by every point over the sequence
        mean_h = np.mean(all_h, axis=1).reshape((nbr_pts, 1))
        mean_w = np.mean(all_w, axis=1).reshape((nbr_pts, 1))

        # Computing the unit norm vectors representing the mean gradient direction for every point
        angles_step_1 = np.mod(np.where(mask_labels > 0, self.mean_gradient_angle_MRI, 0), 180)
        binary_mask = np.where(mask_labels > 0, 1, 0).astype(float)
        kernel = np.ones((3, 3), np.float32)
        active_pixels = cv2.filter2D(binary_mask, -1, kernel)
        grad_sum = cv2.filter2D(angles_step_1, -1, kernel)
        smooth_angles = grad_sum[h, w] / active_pixels[h, w]
        angles = smooth_angles * np.pi / 180.

        v_h = np.cos(angles).reshape((nbr_pts, 1))
        v_w = np.sin(angles).reshape((nbr_pts, 1))

        # Projecting the points on the line that intersects the mean coordinate and that has an angle corresponding
        # to the gradient direction
        dot_prod = (all_h - mean_h) * v_h + (all_w - mean_w) * v_w
        pp_h = dot_prod * v_h
        pp_w = dot_prod * v_w

        # Rotating the data so that we don't need to compute the distances but can simply look at the maximum and
        # minimums taken by a point during the sequence
        cos_rot = np.cos(-angles).reshape((nbr_pts, 1))
        sin_rot = np.sin(-angles).reshape((nbr_pts, 1))
        rot_h = pp_h * cos_rot - pp_w * sin_rot

        # Computing the mean extrema of each interest point
        nbr_extrema = self.external_variables["nbr_extrema_for_mean_amplitude_distance"]
        max_extrema_mean = np.mean(np.partition(rot_h, -nbr_extrema, axis=1)[:, -nbr_extrema:], axis=1)
        min_extrema_mean = np.mean(-np.partition(-rot_h, -nbr_extrema, axis=1)[:, -nbr_extrema:], axis=1)
        distances = max_extrema_mean - min_extrema_mean

        max_distances_mask1 = np.zeros((self.image_height, self.image_width))
        max_distances_mask1[h, w] = distances

        if self.show_figures and show_projected_distance:
            figure = plt.figure(figsize=(16, 10), dpi=80)
            figure.add_subplot(1, 2, 1)
            plt.imshow(max_distances_mask1, cmap=plt.cm.gray)
            plt.axis("off")
            plt.title("Maximum projected distance")

        res = []
        for region in regionprops(mask_labels):
            hh, ww = region.coords.T
            idx = np.argmax(max_distances_mask1[hh, ww])
            idx2 = coo_2_idx[(hh[idx], ww[idx])]
            p_mid = np.array([mean_h[idx2], mean_w[idx2]])
            amp_max = max_extrema_mean[idx2]
            amp_min = min_extrema_mean[idx2]
            # amp_max = np.max(rot_h[idx2, :])
            # amp_min = np.min(rot_h[idx2, :])
            p0 = np.int32(p_mid + np.array([v_h[idx2], v_w[idx2]]) * (amp_min - 2) + 0.5).flatten()
            p1 = np.int32(p_mid + np.array([v_h[idx2], v_w[idx2]]) * (amp_max + 2) + 0.5).flatten()
            res.append((p0, p1, np.int32(p_mid + 0.5).flatten(), (hh[idx], ww[idx])))

        self.VideoViewer.add_layer('navigators', res, color='yellow')
        return res

    def is_edge_in_ct(self, region, show_result=False):
        h, w = region.coords[..., 0], region.coords[..., 1]
        h_min, w_min, h_max, w_max = [min(h), min(w), max(h), max(w)]
        hist_2d, _, _ = np.histogram2d(self.mri_images[0][h_min:h_max, w_min:w_max].ravel(),
                                       self.ct_mi_mri_seq[0][h_min:h_max, w_min:w_max].ravel(), bins=20)
        mi = Tools.mutual_information(hist_2d)
        res = [(0, mi, (h_min, w_min, h_max, w_max))]
        for img_number, (h_new, w_new) in enumerate(self.Flow_Memory.get_coordinates_after_flow(coordinates=(h, w))):
            if img_number == len(self.mri_images) - 1:
                break
            h_min, w_min, h_max, w_max = [min(h_new), min(w_new), max(h_new), max(w_new)]
            hist_2d, _, _ = np.histogram2d(self.mri_images[img_number + 1][h_min:h_max, w_min:w_max].ravel(),
                                           self.ct_mi_mri_seq[img_number + 1][h_min:h_max, w_min:w_max].ravel(),
                                           bins=20)
            res.append((img_number + 1, Tools.mutual_information(hist_2d), (h_min, w_min, h_max, w_max)))
        res = sorted(res, key=lambda i: i[1], reverse=True)
        if self.show_figures and show_result:
            h_min, w_min, h_max, w_max = res[0][2]
            mri_img = cv2.cvtColor(self.mri_images[res[0][0]], cv2.COLOR_GRAY2RGB)
            ct_img = cv2.cvtColor(self.ct_mi_mri_seq[res[0][0]], cv2.COLOR_GRAY2RGB)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax = axes.ravel()
            ax[0].imshow(cv2.rectangle(mri_img, (w_min, h_min), (w_max, h_max), (255, 0, 0), 2))
            ax[0].set_title("MRI Image")
            ax[1].imshow(cv2.rectangle(ct_img, (w_min, h_min), (w_max, h_max), (255, 0, 0), 2))
            ax[1].set_title("CT Image")
            y = [elem[1] for elem in res]
            x = np.arange(len(res))
            mean = [np.mean(y[:i]) for i in range(len(y))]
            ax[2].plot(x, y, color='blue')
            ax[2].plot(x, mean, color='red')
            ax[2].set_title("Decreasing Mutual Information")
            ax[2].set_xlabel("Images")
            ax[2].set_ylabel("Mutual Information")
            plt.show()
        return True

    def show_movement_direction_change(self):
        if self.number_changements is None:
            return
        else:
            X = np.arange(len(self.number_changements))
            plt.figure(figsize=(20, 5))
            plt.plot(X, np.array(self.number_changements), '.--g')
            plt.show()

    def export_video(self, file_path, slower=1):
        self.VideoViewer.create_video(file_path, slower)

    def vv_activate_layer(self, layername):
        self.VideoViewer.activate_layer(layername)

    def vv_deactivate_layer(self, layername):
        self.VideoViewer.deactivate_layer(layername)

    def vv_change_layer_opacity(self, layername, opacity):
        self.VideoViewer.set_layer_opacity(layername, opacity)

    def vv_change_flow_spacing(self, spacing):
        self.VideoViewer.change_flow_spacing(spacing)

    def vv_change_flow_vector_length(self, vector_length):
        self.VideoViewer.change_flow_vector_length(vector_length)

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- Auxiliary Functions ----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def bring_in_interval(data, low, high, IQR=100, show_histograms=False):
        """
        :param data: a list of numpy arrays
        :param low: float corresponding to the lower bound of the interval wanted
        :param high: float corresponding to the higher bound of the interval wanted
        :param IQR: float between 0 and 100 higher percentage of points that will be clipped
        :param show_histograms: boolean value, True if we want to show the mapping of the first array of the list
        :return: a list of numpy arrays with the data brought back in a wanted interval
        """
        assert 0 <= IQR <= 100
        if type(data) != list:
            data = [data]
        data_scaled = [(arr - np.mean(arr)) / np.std(arr) for arr in data]
        data_scaled = [arr - np.min(arr) for arr in data_scaled]
        data_scaled = [np.clip(arr, 0, iqr(arr, rng=(0, IQR))) for arr in data_scaled]
        size = high - low
        data_scaled = [arr / np.max(arr) * size + low for arr in data_scaled]
        if show_histograms:
            number_intervals = 30
            maximum = np.max(data[0])
            interval = (maximum + number_intervals) / number_intervals
            plt.subplot(121)
            _ = plt.hist(data[0], bins=np.arange(0, interval * number_intervals, interval))
            plt.subplot(122)
            size_interval = np.max(data_scaled[0]) - np.min(data_scaled[0])
            _ = plt.hist(data_scaled[0], bins=np.arange(np.min(data_scaled[0]), np.max(data_scaled[0]),
                                                        size_interval / number_intervals))
            plt.show()
        if len(data_scaled) == 1:
            return data_scaled[0]
        return data_scaled

    @staticmethod
    def get_coordinate_pixel_move_most(distances, grid, ratio):
        n = int(ratio * grid.number_points)
        number_images = distances.shape[0]
        mean_distances_per_pixel = np.sum(distances, axis=0) / number_images
        top_n_index = np.argpartition(mean_distances_per_pixel, -n)[-n:]
        top_n_coordinates = [(grid.semi_spacing + grid.spacing * (i % grid.width_number_elements),
                              grid.semi_spacing + grid.spacing * (i // grid.width_number_elements)) for i in
                             top_n_index]
        return top_n_coordinates

    @staticmethod
    def compute_flow(images, method='farneback', interval=1):
        """Returns a list of the size of Images minus the interval.
        Each element of the list is a 3-D array where the two first dimensions correspond to the pixels and the third
        is of size 2 and contains respectively the width_flow and the height_flow.
        """
        if method == 'farneback':
            flows = [cv2.calcOpticalFlowFarneback(prev=images[i], next=images[i + interval], flow=None,
                                                  pyr_scale=0.5, levels=1, winsize=7,
                                                  iterations=3, poly_n=5, poly_sigma=1.1, flags=0) for i in
                     range(len(images) - interval)]
        else:
            assert False, "Method must equal 'farneback'. Implement other methds !!"
        return flows

    @staticmethod
    def ratio_diff_elem(a, b):
        assert (a.shape == b.shape)
        return np.count_nonzero(a - b) / a.size

    def get_outbound_rectangles(self, top_n_coordinates, grid):
        top_n_x = np.unique(np.array([x for x, _ in top_n_coordinates]))
        joint_top_n_x = self.group_consecutive(top_n_x, grid.spacing)
        dico = {}
        number_groups = len(joint_top_n_x)
        ret = []
        for i in range(number_groups):
            dico[i] = {}
        for x, y in top_n_coordinates:
            for i in range(number_groups):
                if y not in dico[i] and x in joint_top_n_x[i]:
                    dico[i][y] = True
                    break
        for i in range(number_groups):
            keys = list(dico[i].keys())
            keys.sort()
            joint_top_n_y = self.group_consecutive(keys, grid.spacing)
            for j in range(len(joint_top_n_y)):
                ret.append([joint_top_n_x[i][0] - grid.semi_spacing, joint_top_n_x[i][-1] + grid.semi_spacing,
                            joint_top_n_y[j][0] - grid.semi_spacing, joint_top_n_y[j][-1] + grid.semi_spacing])
        return ret

    @staticmethod
    def group_consecutive(list_to_process, step, thresh=5):
        run = []
        result = [run]
        expect = None
        for elem in list_to_process:
            if (elem == expect) or (expect is None):
                run.append(elem)
            else:
                run = [elem]
                result.append(run)
            expect = elem + step
        ret = [elem for elem in result if len(elem) > thresh]
        return ret
