import numpy as np
from scipy.spatial.transform import Rotation as R


q = [0.1946025136148841, 0.2883167955284461,
     0.324667150317451, 0.8795423874160202]


def extract_rotation(quaternions):
    res = R.from_quat(quaternions).apply([1, 0, 0])

    return np.arctan2(res[1], res[0])


print(
    extract_rotation(q)
)
