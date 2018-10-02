# graph2net

A notebook for building models via stacking of convolutional cells. The architecture of each cell can specified by a connectivity matrix, thus reducing the problem of network design to that of simply designing a connectivity matrix.

### The Cell Matrix
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

### Examples
Thus, a simple VGGBlock can be loosely specified by:
| 0 | 10 | 0  | 0 |
| 0 | 0  | 10 | 0 |
| 0 | 0  | 0  | 3 |
| 0 | 0  | 0  | 0 |
![VGGBlock](https://github.com/RobGeada/graph2net/blob/master/images/vggblock.png)

Thus, a single ResBlock can be loosely specified by:
| 0 | 11 | 1  | 0 |
| 0 | 0  | 11 | 0 |
| 0 | 0  | 0  | 3 |
| 0 | 0  | 0  | 0 |
or 
![ResBlock](https://github.com/RobGeada/graph2net/blob/master/images/resblock.png)

Alternatively, we can just generate some random matrices and see what networks they generate:
![RandBlock1](https://github.com/RobGeada/graph2net/blob/master/images/randblock1.png)
![RandBlock2](https://github.com/RobGeada/graph2net/blob/master/images/randblock2.png)
![RandBlock3](https://github.com/RobGeada/graph2net/blob/master/images/randblock3.png)
