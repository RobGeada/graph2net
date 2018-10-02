# graph2net

A notebook for building models via stacking of convolutional cells. The architecture of each cell can specified by a connectivity matrix, thus reducing the problem of network design to that of simply designing a connectivity matrix.

## The Cell Matrix
The cell matrix is a connectivity matrix, specifying what kind of operation connects node *i* to node *j*. Available operations derive from the [NASNet Search Space](https://arxiv.org/pdf/1707.07012.pdf) and are specified via index, as follows:
0) zero
1) identity
2) 3x3 average pooling
3) 3x3 max pooling
4) 5x5 max pooling
5) 7x7 max pooling
6) 1x7 followed by 7x1 Convolution
7) 1x3 followed by 3x1_Convolution
8) Dilated Convolution_3x3
9) Convolution_1x1
10) Convolution_3x3
11) Depthwise Separable Convolution_3x3
12) Depthwise Separable Convolution_5x5
13) Depthwise Separable Convolution_7x7

Cell matrices must be *strictly upper triangular*, thus corresponding to a DAG and preventing recursive behavior. Node 0 is the cell input, whereas the final node corresponds to cell output. Therefore, any connection in the first row of the cell matrix indicates the selected node receieves cell input, wheras any value in the final column of the cell matrix indicates the node sends its output to the cell output. See below for some examples.

## Cell Matrix Examples
Thus, a simple VGGBlock can be loosely specified by:

||node0|node1|node2|node3|
|-|-|-|-|-|
| **node0** | 0 | 10 | 0  | 0 |
| **node1** | 0 | 0  | 10 | 0 |
| **node2** | 0 | 0  | 0  | 3 |
| **node3** | 0 | 0  | 0  | 0 |

![VGGBlock](https://github.com/RobGeada/graph2net/blob/master/images/vggblock.png)

A single ResBlock can be loosely specified by:

||node0|node1|node2|node3|
|-|-|-|-|-|
| **node0** | 0 | 11 | 1  | 0 |
| **node1** | 0 | 0  | 1  | 0 |
| **node2** | 0 | 0  | 0  | 3 |
| **node3** | 0 | 0  | 0  | 0 |

![ResBlock](https://github.com/RobGeada/graph2net/blob/master/images/resblock.png)

## Random Networks
Alternatively, we can just generate some random matrices and see what cells they generate:
![RandBlock1](https://github.com/RobGeada/graph2net/blob/master/images/randblock1.png)
![RandBlock2](https://github.com/RobGeada/graph2net/blob/master/images/randblock2.png)
![RandBlock3](https://github.com/RobGeada/graph2net/blob/master/images/randblock3.png)

## Cell Rules
To ensure compatibility of tensors and that cells scale data correctly, the following rules are implemented:
* Any connection from cell input doubles the number of channels via 1x1 convolutions
* Any connection to cell output halves the spatial dimensions via a stride-2 operation

## Usage
See the notebook for usage.

## Future Plans
* Testing of random cell networks
* Conversion to CUDA
* Evolution of cell matrices to find more optimal networks
