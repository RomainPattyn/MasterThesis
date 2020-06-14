import importlib
import os
import time
from sys import stdout

import FlowMemory
import Grid
import Tools
import cv2
import numpy as np

importlib.reload(Grid)
importlib.reload(Tools)


class VideoViewer:
    def __init__(self, images, Flow_Memory):
        self.parameters = {}
        self.active_layers = {}
        self.opacity = {}
        self.color = {}
        self.Images = images

        self.image_height, self.image_width = self.Images[0].shape

        if self.Images[0].ndim == 2:
            self.Images = [np.ndarray.astype(img, "uint8") for img in self.Images]
            self.Images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in self.Images]

        self.colorDico = {'blue': (200, 0, 0), 'green': (0, 200, 0), 'red': (0, 0, 200), 'white': (255, 255, 255),
                          'yellow': (255, 255, 0), 'light_green': (47, 255, 173)}
        self.Flow_Memory = Flow_Memory

    def change_flow_spacing(self, spacing):
        assert "flow" in self.parameters
        self.parameters["flow"][0] = spacing

    def change_flow_vector_length(self, vector_length):
        assert "flow" in self.parameters
        self.parameters["flow"][1] = vector_length

    def set_layer_opacity(self, layername, opacity):
        if layername not in self.parameters:
            assert False, "ERROR : Trying to change the opacity of an innexistant layer. (layer : {0})".format(
                layername)
        if layername not in self.active_layers:
            print("WRANING : Changing layer opacity while layer is inactive. (layer : {0})".format(layername))
        else:
            self.opacity[layername] = opacity

    def set_layer_color(self, layername, color):
        if layername not in self.parameters:
            assert False, "ERROR : Trying to change the color of an innexistant layer. (layer : {0})".format(layername)
        if layername not in self.active_layers:
            print("WRANING : Changing layer color while layer is inactive. (layer : {0})".format(layername))
        else:
            self.color[layername] = color

    def deactivate_layer(self, layername):
        if layername not in self.active_layers:
            print("WRANING : Trying to deactivate an inactive layer. (layer : {0})".format(layername))
        else:
            del self.active_layers[layername]

    def activate_layer(self, layername):
        if layername not in self.parameters:
            assert False, "ERROR : Trying to activate an innexistant layer. (layer : {0})".format(layername)
        elif layername in self.active_layers:
            print("WRANING : Trying to activate a layer that is already active. (layer : {0})".format(layername))
        else:
            self.active_layers[layername] = True

    def add_layer(self, layername, content, opacity=0.5, color='green'):
        if layername == 'following_points':
            if layername not in self.parameters:
                self.parameters[layername] = {}
                self.parameters[layername][0] = content
                self.color[layername] = {}
                self.color[layername][0] = color
                self.opacity[layername] = opacity
                self.active_layers[layername] = True
            else:
                self.parameters[layername][len(self.parameters[layername])] = content
                self.color[layername][len(self.color[layername])] = color
        else:
            if layername in self.parameters:
                print('WARNING : Overriding layer : ' + layername + '.')
            self.parameters[layername] = content
            self.opacity[layername] = opacity
            self.active_layers[layername] = True
            self.color[layername] = color

    def create_video(self, file_path, slower=1):
        """Create a video from a list of images.
        If the images are in black and white, the method first converts them to RGB.
        """
        processed_image_list = self.process_layers()
        if "ct_correspondance" in self.active_layers:
            data = self.parameters["ct_correspondance"]
            assert (len(data) == len(processed_image_list[0]))
            if data[0].ndim == 2:
                data = [np.ndarray.astype(img, "uint8") for img in data]
                data = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in data]
            if "navigators" in self.active_layers and self.active_layers["navigators"]:
                navig_pts = self.parameters["navigators"]
                color_navig = self.colorDico[self.color["navigators"]]
                data = self.draw_navigators_centroid_ellipse(data, navig_pts, color_navig)
            processed_image_list.append(data)
        if len(processed_image_list) > 1:
            processed_image_list = self.merge_images(processed_image_list)
        else:
            processed_image_list = processed_image_list[0]
        if os.path.exists(file_path):
            print("WARNING : Overriding file : " + file_path)
            os.remove(file_path)

        # If the image is in black and white we need to convert it into "RGB" to create the video
        if len(processed_image_list[0].shape) == 2:
            processed_image_list = [np.ndarray.astype(img, "uint8") for img in processed_image_list]
            processed_image_list = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in processed_image_list]

        height, width = processed_image_list[0].shape[0:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, 20.0, (width, height))
        for frame in processed_image_list:
            for _ in range(slower):
                out.write(frame)
                cv2.imshow('video', frame)
        out.release()

    # TODO :: crop images to be same length and Content of dictionary too
    def process_layers(self):
        return_images = [[img.copy() for img in self.Images]]
        for layer in self.active_layers:
            data = self.parameters[layer]
            opacity = self.opacity[layer]
            additional_layer = None
            if layer == 'flow':
                # data = (grid_spacing, flow_norm_multiplier)
                color = self.colorDico[self.color[layer]]
                grid_spacing = data[0]
                grid = Grid.Grid(grid_spacing, self.image_height, self.image_width)
                additional_layer = self.draw_flow(return_images, self.Flow_Memory.flow_consecutives, grid, data[1],
                                                  color=color)
            elif layer == 'cluster_bounds':
                color = self.colorDico[self.color[layer]]
                # data = [ ..., (cluster_i_x_min, cluster_i_x_max, cluster_i_y_min, cluster_i_y_max), ... ]
                additional_layer = self.draw_cluster(return_images, data, color=color)
            elif layer == 'cluster_points':
                color = self.colorDico[self.color[layer]]
                # data = ([ ..., (interest_point_i_x,interest_point_i_y), ... ], square_size_around_interest_point)
                additional_layer = self.draw_cluster(return_images, data, color=color)
            elif layer == 'interest_points':
                color = self.colorDico[self.color[layer]]
                # data = ([ ..., (interest_point_i_x,interest_point_i_y), ... ], square_size_around_interest_point)
                additional_layer = self.draw_interest_points(return_images, data, color=color)
            elif layer == 'square_lines':
                assert len(return_images) == 1
                return_images = [img.copy() for img in return_images[0]]
                additional_layer = []
                color = self.colorDico[self.color[layer][i]]
                space = data[0]
                additional_layer.append(self.draw_follow_point(return_images, data[i], color=color))
            elif layer == 'following_points':
                assert len(return_images) == 1
                number_layers = len(data)
                # data = [[(w1, h1), (w2, h2), ... ], [ ... ], [ ... ] ]
                return_images = [[img.copy() for img in return_images[0]] for _ in range(number_layers)]
                additional_layer = []
                for i in range(number_layers):
                    color = self.colorDico[self.color[layer][i]]
                    if data[i].size > 0:
                        additional_layer.append(self.draw_follow_point(return_images[i], data[i], color=color))
                    else:
                        additional_layer.append(return_images[i])
                if "navigators" in self.active_layers and self.active_layers["navigators"]:
                    navig_pts = self.parameters["navigators"]
                    color_navig = self.colorDico[self.color["navigators"]]
                    additional_layer[-1] = self.draw_navigators_centroid_ellipse(additional_layer[-1], navig_pts,
                                                                                 color_navig)
            elif layer == 'navigators':
                continue
            elif layer == 'ct_correspondance':
                continue
            else:
                print("ERROR : Layer '{0}' unkown.".format(layer))
                assert False
            return_images = [[cv2.addWeighted(img_with_layer, opacity, img, 1 - opacity, 0) for img_with_layer, img in
                              zip(additional_layer_sub, return_images_sub)] for additional_layer_sub, return_images_sub
                             in zip(additional_layer, return_images)]
        return return_images

    def draw_follow_point(self, images, coordinates, color):
        additional_layer = [img.copy() for img in images]
        h, w = coordinates.T
        additional_layer[0][h, w] = color
        for i, (h_new, w_new) in enumerate(self.Flow_Memory.get_coordinates_after_flow(coordinates=(h, w))):
            additional_layer[i + 1][h_new, w_new] = color
        return additional_layer

    def draw_navigators_projection(self, additional_layer, pts, color):
        for p0, p1, pmid, pf in pts:
            additional_layer = [cv2.line(img, (p0[1], p0[0]), (p1[1], p1[0]), color, 1) for img in additional_layer]
            additional_layer = [cv2.circle(img, (pmid[1], pmid[0]), 1, (0, 0, 255), -1) for img in additional_layer]
            additional_layer = [cv2.circle(img, (pf[1], pf[0]), 1, (0, 255, 0), -1) for img in additional_layer]
        return additional_layer

    def draw_navigators_centroid_ellipse(self, additional_layer, pts, color):
        for p0, p1 in pts:
            additional_layer = [cv2.line(img, (p0[0], p0[1]), (p1[0], p1[1]), color, 1) for img in additional_layer]
        return additional_layer

    @staticmethod
    def merge_images(images_list):
        img_ref = images_list[0]
        for images in images_list[1:]:
            img_ref = [np.concatenate((img_r, img), axis=1) for img_r, img in zip(img_ref, images)]
        return img_ref

    @staticmethod
    def draw_grid(images, w_grid_coordinates, h_grid_coordinates):
        def draw_grid_single_img(img, x, y):
            img_layer = img.copy()
            for (x1, y1) in zip(x, y):
                cv2.circle(img_layer, (x1, y1), 1, (168, 168, 168), -1)
            return img_layer

        additional_layer = [img.copy() for img in images]
        return [draw_grid_single_img(img, w_grid_coordinates, h_grid_coordinates) for img in additional_layer]

    @staticmethod
    def draw_flow(images, flows, grid, vector_norm_multiplier, color):
        def draw_flow_single_img(img, flow, h, w):
            fh, fw = zip(*flow[h, w] * vector_norm_multiplier)
            lines = np.vstack([w, h, w + fw, h + fh]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)  # Round to upper bound
            img_layer = img.copy()
            cv2.polylines(img_layer, lines, 0, color)
            return img_layer

        additional_layer = [[img.copy() for img in images_sub] for images_sub in images]
        h_grid_coordinates = grid.height_coordinates
        w_grid_coordinates = grid.width_coordinates
        flows.insert(0, np.zeros(flows[0].shape))
        return [[draw_flow_single_img(img, flow, h_grid_coordinates, w_grid_coordinates) for img, flow in
                 zip(additional_layer_sub, flows)] for additional_layer_sub in additional_layer]

    @staticmethod
    def draw_cluster_points(images, data, color):
        additional_layer = [[img.copy() for img in images_sub] for images_sub in images]
        semi_square_length = data[1] // 2
        for x, y in data[0]:
            additional_layer = [[cv2.rectangle(img, (x - semi_square_length, y - semi_square_length),
                                               (x + semi_square_length, y + semi_square_length), color, -1) for img in
                                 additional_layer_sub] for additional_layer_sub in additional_layer]
        return additional_layer

    @staticmethod
    def draw_interest_points(images, data, color):
        additional_layer = [[img.copy() for img in Images_Sub] for Images_Sub in images]
        for x, y in data[0]:
            additional_layer[x, y] = color
        return additional_layer

    @staticmethod
    def draw_cluster(images, data, color):
        additional_layer = [[img.copy() for img in images_sub] for images_sub in images]
        for x_min, x_max, y_min, y_max in data:
            additional_layer = [
                [cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1) for img in additional_layer_sub] for
                additional_layer_sub in additional_layer]
            """
            #  Draws a horizontal line at the middle of the square. 
            mid_point = (y_min+y_max)//2
            additional_layer = [
                [cv2.line(img, (x_min, mid_point), (x_max, mid_point), color, 1) for img in additional_layer_sub] for
                additional_layer_sub in additional_layer]
            """
        return additional_layer
