import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from sys import stdout
from sklearn.cluster import MeanShift, KMeans


def div_zero(n, d):
    return n / d if d else 0


def print_loading_bar(img_number, LOADING_BAR_CST):
    if img_number % LOADING_BAR_CST == 0:
        stdout.write('.')
        stdout.flush()


def show_hough_transform(image, probabilistic, n_clusters):
    if probabilistic:
        lines = probabilistic_hough_line(image, threshold=10, line_length=5, line_gap=3)
        # Generating figure 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Mask of interest points')

        ax[1].imshow(image * 0)
        X = []
        for line in lines:
            p0, p1 = line
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
            X.append([p0[0], p1[0], p0[1], p1[1]])
        X = np.array(X)
        ax[1].set_xlim((0, image.shape[1]))
        ax[1].set_ylim((image.shape[0], 0))
        ax[1].set_title('Probabilistic Hough')
        clustering = KMeans(n_clusters=n_clusters + 5).fit(X)
        print(len(lines))
        print(len(clustering.cluster_centers_))
        ax[2].imshow(image * 0)
        for p00, p10, p01, p11 in clustering.cluster_centers_:
            ax[2].plot((p00, p10), (p01, p11))
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_title('Mean Shhift 4D clustering')
        plt.tight_layout()
        plt.show()

    else:
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(image, theta=tested_angles)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title('Mask of interest points')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + h),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                     cmap="gray", aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap="gray")
        origin = np.array((0, image.shape[1]))
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[2].plot(origin, (y0, y1), '-r')
        ax[2].set_xlim(origin)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        plt.tight_layout()
        plt.show()


def bounding_box(h, w):
    return [(min(h), min(w)), (max(h), max(w))]


def has_enough_neighbors(coordinate, half_window, threshold, coordinates):
    neighbors = 0
    for x_shift in range(- half_window, half_window + 1):
        for y_shift in range(- half_window, half_window + 1):
            if (coordinate[0] + x_shift, coordinate[1] + y_shift) in coordinates:
                neighbors += 1
                if neighbors > threshold:
                    return True
    return False


def filter_salt(coordinates, k_size):
    assert (k_size % 2 == 1 and k_size > 0)
    result = []
    half_window = k_size // 2
    threshold = int(k_size ** 2 / 6 + 1)
    t_start = time.time()
    for coo in coordinates:
        if has_enough_neighbors(coo, half_window, threshold, coordinates):
            result.append(coo)
    t_end = time.time()
    print("Time of list method : ", t_end - t_start)
    print("Efficiency : ", len(coordinates), "\t", len(result), "\t", len(result) / len(coordinates))
    return result


def filter_median_blur(coordinates, ref_img, ratio, k_size):
    blured_img = cv2.medianBlur(ref_img, 7)
    gradient_in_x = cv2.Sobel(blured_img, cv2.CV_32F, 1, 0, ksize=k_size)
    gradient_in_y = cv2.Sobel(blured_img, cv2.CV_32F, 0, 1, ksize=k_size)
    magnitude = cv2.magnitude(gradient_in_x, gradient_in_y)
    coo_and_mag = [(top_coo, magnitude[top_coo[1], top_coo[0]]) for top_coo in coordinates]
    coo_and_mag.sort(key=lambda tup: tup[1], reverse=True)
    amount = int(ratio * len(coo_and_mag))
    return [coo for coo, _ in coo_and_mag[:amount]]


def get_region_mean_histogram_variance(region, histograms, width):
    cumulative_variance = 0.
    for x, y in region.coords:
        cumulative_variance += np.var(histograms[x * width + y])
    return cumulative_variance / len(region.coords)


def get_region_edge_value(region, ref_img, k_size):
    blured_img = cv2.GaussianBlur(ref_img, (5, 5), 0)
    plt.imshow(blured_img, cmap="gray")
    gradient_in_x = cv2.Sobel(blured_img, cv2.CV_32F, 1, 0, ksize=k_size)
    gradient_in_y = cv2.Sobel(blured_img, cv2.CV_32F, 0, 1, ksize=k_size)
    magnitude = cv2.magnitude(gradient_in_x, gradient_in_y)
    cumulative_edge_magnitude = 0.
    for x, y in region.coords:
        cumulative_edge_magnitude += magnitude[x, y]
    return cumulative_edge_magnitude / len(region.coords)


def mutual_information(hgram):
    """ Mutual information for joint histogram
        Code from https://matthew-brett.github.io/teaching/mutual_information.html
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def re_target_coo(array, shift, dimension):
    H, W = dimension
    shift_h, shift_w = shift
    result = array - (shift_w, shift_h)
    delta_width = array[:, :, 1] * (W - shift_w - result.shape[1] - 1)
    # result[:, :, 0] -= delta_width
    return result


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def get_histograms(images, angles, gradient_magnitudes, max_angle, n_bins, bounds, fixed_point=False,
                   normalize=False):
    assert (len(angles) == len(gradient_magnitudes))
    ref_img = images[0]
    height = ref_img.shape[0]
    width = ref_img.shape[1]
    interval_list = list(np.linspace(0, max_angle, n_bins + 1, endpoint=True))
    assert (len(interval_list) - 1 == n_bins)
    interval_list[-1] += 1  # To include the value for an angle of max_angle
    counts = [np.logical_and(angles[0] >= interval_list[index], angles[0] < interval_list[index + 1]).astype(int)
              for index in range(n_bins)]
    histograms = [count * gradient_magnitudes[0] for count in counts]
    for img_number, (img, a_e, g_m) in enumerate(zip(images[1:], angles[1:], gradient_magnitudes[1:])):
        assert (np.max(a_e) <= max_angle)
        if img_number % LOADING_BAR_CST == 0:
            stdout.write('.')
            stdout.flush()
        if fixed_point:
            new_counts = [np.logical_and(a_e >= interval_list[index], a_e < interval_list[index + 1]).astype(int)
                          for index in range(n_bins)]
            histograms = [hist + count * g_m for hist, count in zip(histograms, new_counts)]
        else:
            pos_index = self.FlowMemory.get_coordinates_after_flow_index(img_number, bounds=bounds)
            new_counts = [np.logical_and(a_e >= interval_list[index], a_e < interval_list[index + 1]).astype(int)
                          [pos_index] for index in range(n_bins)]
            histograms = [hist + count * g_m[pos_index] for hist, count in zip(histograms, new_counts)]
        counts = [count + new_count for count, new_count in zip(counts, new_counts)]
    stdout.write('\n')
    if normalize:
        histograms = [np.divide(hist, count, out=np.zeros_like(hist), where=count != 0) for hist, count in
                      zip(histograms, counts)]
    hist_array = np.array(histograms)
    return hist_array.transpose().reshape((height, width, n_bins))


def process_histograms(histograms, masks, ref_img, max_angle_error, color):
    def get_class(point):
        if point[0] >= threshold:
            return 0
        else:
            return 1

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    threshold = height // 2
    n_bins = histograms.shape[1]
    number_classes = 2
    mean_hist_per_class = np.zeros((number_classes, n_bins))
    counters = np.zeros(number_classes)
    cumulative_variance = np.zeros(number_classes)

    for i, mask in enumerate(masks):
        for x, y in np.argwhere(mask == 1):
            point_class = get_class((x, y))
            if i == 2:
                point_class = 1
            counters[point_class] += 1
            mean_hist_per_class[point_class] += histograms[x * width + y]
            cumulative_variance[point_class] += np.var(histograms[x * width + y])

    x_axis = np.linspace(0, max_angle_error, n_bins, endpoint=False)
    plot_width = max_angle_error * 8 / (n_bins * 10)
    for index in range(number_classes):
        if counters[index] != 0:
            mean_hist_per_class[index] /= counters[index]
            cumulative_variance[index] /= counters[index]
    y_max = np.max(mean_hist_per_class) * 11 / 10

    print("\tClass Label 0 = > point in lower half (good)")
    print("\tClass Label 1 = > point in upper half (bad)")
    print("\tMeans computed with : ", int(np.sum(counters)), " samples.")

    plt.figure()
    for index in range(number_classes):
        print(" \t\t- ", int(counters[index]), "points of class label ", index, ". Which has a mean variance of ",
              round(cumulative_variance[index], 3), ".")
        sub_plot_number = 100 + number_classes * 10 + index + 1
        plt.subplot(sub_plot_number)
        plt.bar(x_axis, mean_hist_per_class[index], plot_width, color=color)
        plt.ylim([0, y_max])
        plt.gca().set_title("Class " + str(index))
    plt.show()
