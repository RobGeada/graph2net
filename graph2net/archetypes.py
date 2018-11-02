from graph2net.graph_generators import build_cell

vggNet = build_cell([
    [0, 1, 'conv_3x3'],
    [1, 2, 'conv_3x3'],
    [2, 3, 'conv_3x3'],
    [3, 4, 'max_pool_3x3']
])

resNet = build_cell([
    [0, 1, 'conv_3x3'],
    [0, 2, 'identity'],
    [1, 2, 'conv_3x3'],
    [2, 3, 'conv_3x3'],
    [2, 4, 'identity'],
    [3, 4, 'conv_3x3'],
    [4, 5, 'avg_pool_3x3']
])

resNeXt = build_cell([
    [0, 1, 'halve_channel'],
    [1, 2, 'sep_conv_3x3'],
    [2, 3, 'double_channel'],
    [0, 3, 'identity'],
])

inception = build_cell([
    [0, 4, 'conv_1x1'],
    [0, 1, 'conv_1x1'],
    [1, 4, 'sep_conv_3x3'],
    [0, 2, 'conv_1x1'],
    [2, 4, 'sep_conv_5x5'],
    [0, 3, 'max_pool_3x3'],
    [3, 4, 'conv_1x1'],
])
