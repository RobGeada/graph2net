import numpy as np

# 0: 'zero',
# 1: 'identity'
# 2: 'double_channel'
# 3: 'halve_channel'
# 4: 'avg_pool_3x3'
# 5: 'max_pool_3x3'
# 6: 'max_pool_5x5'
# 7: 'max_pool_7x7'
# 8: '1x7_7x1_conv'
# 9: '1x3_3x1_conv'
# 10: 'dil_conv_3x3'
# 11: 'conv_1x1'
# 12: 'conv_3x3'
# 13: 'sep_conv_3x3'
# 14: 'sep_conv_5x5'
# 15: 'sep_conv_7x7'


resNet = np.array([[0, 12, 1, 0, 0],
                   [0, 0, 12, 0, 0],
                   [0, 0, 0, 12, 1],
                   [0, 0, 0, 0, 12],
                   [0, 0, 0, 0, 0]])

resNeXt = np.array([[0, 3, 0, 1],
                    [0, 0, 13, 0],
                    [0, 0, 0, 2],
                    [0, 0, 0, 0]])

inception = np.array([[0, 11, 11, 5, 11],
                      [1, 0, 0, 0, 13],
                      [1, 0, 0, 0, 14],
                      [1, 0, 0, 0, 11],
                      [1, 0, 0, 0, 1]])

