import cv2
import numpy as np

from utils import utils


class FakeSimulator:
    def __init__(self, cfg, world):
        # world configuration
        self.world = world
        self.gsd = cfg["gsd"]  # m/pixel
        self.world_range = cfg["world_range"]  # pixel

        # sensor configuration
        self.sensor_resolution = cfg["sensor"]["resolution"]
        self.sensor_angle = cfg["sensor"]["angle"]

    def get_measurement(self, pose):
        fov_info = utils.get_fov(pose, self.sensor_angle, self.gsd, self.world_range)

        fov_corner, range_list = fov_info
        gsd = [
            (np.linalg.norm(fov_corner[1] - fov_corner[0])) / self.sensor_resolution[0],
            (np.linalg.norm(fov_corner[3] - fov_corner[0])) / self.sensor_resolution[1],
        ]
        rgb_image_raw = self.world[range_list[2] : range_list[3], range_list[0] : range_list[1], :]
        rgb_image = cv2.resize(rgb_image_raw, tuple(self.sensor_resolution))

        return {"image": rgb_image, "fov": fov_corner, "gsd": gsd}
