"""Test some of the functions in nonlinear_nisl_visualisation"""

import numpy as np
from nonlinear_nisl_visualisation import create_upsampled_grid

def test_upsampling():
    """ """
    data = np.array(((0, 1, 2), (2, 3, 4) ))
    expected_upsampled_data = np.array(((0, 0.5, 1, 1.5, 2, 1), (1, 1.5, 2, 2.5, 3, 2),
                                        (2, 2.5, 3, 3.5, 4, 3), (1, 1.5, 2, 2.5, 3, 2)))
    upsampled_data = create_upsampled_grid(data)
    for yidx in [0,1]:
        for xidx in [0,1,2]:
            assert(expected_upsampled_data[yidx, xidx] == upsampled_data[yidx, xidx])
    #assert all([a == b for a, b in zip(upsampled_data, expected_upsampled_data)])
    #assert(expected_upsampled_data==upsampled_data)


#if __name__ == "__main__":
#    test_upsampling()
