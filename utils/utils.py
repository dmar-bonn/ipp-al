from typing import List

import numpy as np

LABELS = {
    "scw": {
        "soil": {"color": (0, 0, 0), "id": 0},
        "crop": {"color": (0, 255, 0), "id": 1},
        "weed": {"color": (255, 0, 0), "id": 2},
    },
    "potsdam": {
        "boundary line": {"color": (0, 0, 0), "id": 0},
        "imprevious surfaces": {"color": (255, 255, 255), "id": 1},
        "building": {"color": (0, 0, 255), "id": 2},
        "low vegetation": {"color": (0, 255, 255), "id": 3},
        "tree": {"color": (0, 255, 0), "id": 4},
        "car": {"color": (255, 255, 0), "id": 5},
        "clutter/background": {"color": (255, 0, 0), "id": 6},
    },
    "rit18": {
        "bg": {"color": (0, 0, 0), "id": 0},
        "road marking": {"color": (19, 9, 25), "id": 1},
        "tree": {"color": (26, 24, 52), "id": 2},
        "building": {"color": (24, 45, 72), "id": 3},
        "vehicle": {"color": (21, 69, 78), "id": 4},
        "person": {"color": (25, 94, 70), "id": 5},
        "lifeguard chair": {"color": (43, 111, 57), "id": 6},
        "picnic table": {"color": (75, 120, 47), "id": 7},
        "black wood panel": {"color": (114, 122, 49), "id": 8},
        "white wood panel": {"color": (161, 121, 74), "id": 9},
        "landing pad": {"color": (193, 121, 111), "id": 10},
        "buoy": {"color": (209, 128, 156), "id": 11},
        "rocks": {"color": (211, 143, 197), "id": 12},
        "vegetation": {"color": (203, 165, 227), "id": 13},
        "grass": {"color": (194, 193, 242), "id": 14},
        "sand": {"color": (194, 216, 242), "id": 15},
        "lake": {"color": (206, 235, 239), "id": 16},
        "pond": {"color": (229, 247, 240), "id": 17},
        "asphalt": {"color": (255, 255, 255), "id": 18},
    },
}


THEMES = {"weedmap": "scw", "potsdam": "potsdam", "rit18": "rit18"}


def imap2rgb(imap, channel_order, theme):
    """converts an iMap label image into a RGB Color label image,
    following label colors/ids stated in the 'labels' dict.

    Arguments:
        imap {numpy with shape (h,w)} -- label image containing label ids [int]
        channel_order {str} -- channel order ['hwc' for shape(h,w,3) or 'chw' for shape(3,h,w)]
        theme {str} -- label theme

    Returns:
        float32 numpy with shape (channel_order) -- rgb label image containing label colors from dict (int,int,int)
    """
    assert channel_order == "hwc" or channel_order == "chw"
    assert len(imap.shape) == 2
    assert theme in LABELS.keys()

    rgb = np.zeros((imap.shape[0], imap.shape[1], 3), np.float32)
    for _, cl in LABELS[theme].items():  # loop each class label
        if cl["color"] == (0, 0, 0):
            continue  # skip assignment of only zeros
        mask = np.where(imap == cl["id"], 1, 0)
        rgb[:, :, 0] += mask * cl["color"][0]
        rgb[:, :, 1] += mask * cl["color"][1]
        rgb[:, :, 2] += mask * cl["color"][2]
    if channel_order == "chw":
        rgb = np.moveaxis(rgb, -1, 0)  # convert hwc to chw
    return rgb


def toRGB(img, dataset_name):
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
        img = np.argmax(img, axis=-1)

    img = imap2rgb(img, channel_order="hwc", theme=THEMES[dataset_name])
    return img.astype(np.uint8)


def prob_to_logit(prob):
    logit = np.log(prob / (1 - prob))
    return logit


# return 4 corner index of rectangle fov
def get_fov(pose: np.array, sensor_angle: List, gsd: List, world_range: List):
    half_fov_size = pose[2] * np.tan(np.deg2rad(sensor_angle))

    # fov in world coordinate frame
    lu = [pose[0] - half_fov_size[0], pose[1] - half_fov_size[1]]
    ru = [pose[0] + half_fov_size[0], pose[1] - half_fov_size[1]]
    rd = [pose[0] + half_fov_size[0], pose[1] + half_fov_size[1]]
    ld = [pose[0] - half_fov_size[0], pose[1] + half_fov_size[1]]
    corner_list = np.array([lu, ru, rd, ld])

    # fov index in orthomosaic space
    lu_index = [np.floor(lu[0] / gsd).astype(int), np.floor(lu[1] / gsd).astype(int)]
    ru_index = [np.ceil(ru[0] / gsd).astype(int), np.floor(ru[1] / gsd).astype(int)]
    rd_index = [np.ceil(rd[0] / gsd).astype(int), np.ceil(rd[1] / gsd).astype(int)]
    ld_index = [np.floor(ld[0] / gsd).astype(int), np.ceil(ld[1] / gsd).astype(int)]

    index_list = np.array([lu_index, ru_index, rd_index, ld_index])
    min_x = np.min(index_list[:, 0])
    max_x = np.max(index_list[:, 0])
    min_y = np.min(index_list[:, 1])
    max_y = np.max(index_list[:, 1])

    if np.any(np.array([min_x, min_y]) < np.array([0, 0])) or np.any(np.array([max_x, max_y]) > np.array(world_range)):
        raise ValueError("Invalid measurement! Measurement out of environment bounds.")
    else:
        return corner_list, [min_x, max_x, min_y, max_y]
