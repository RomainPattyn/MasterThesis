import numpy as np


class Grid:
    def __init__(self, grid_spacing, image_height, image_width):
        self.spacing = grid_spacing
        self.semi_spacing = grid_spacing // 2
        self.height_coordinates, self.width_coordinates = np.mgrid[self.semi_spacing:image_height:self.spacing,
                                                          self.semi_spacing:image_width:self.spacing].reshape(2,
                                                                                                              -1).astype(
            int)
        self.width_number_elements = (image_width - self.semi_spacing) // self.spacing + 1
        self.height_number_elements = (image_height - self.semi_spacing) // self.spacing + 1
        self.number_points = self.width_number_elements * self.height_number_elements

    def getVectorizedDataOnGrid(self, data):
        """The data is considered to be of size (height, width)
        """
        return data[self.height_coordinates, self.width_coordinates]
