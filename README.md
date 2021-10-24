# Depth estimation from stereo images using Census Transform and Hamming Distance

## General knowledge

Depth estimation refers to the set of algorithms used to obtain a measure of the distance of each point of an image.

This project proposes the use of stereo images which represents a pair of images of the same scene, with a horizontal offset to mimic the left and right eye views. 
Stereo images should provide enough information to extract the depth at which objects are.

The census transform is used to encode each pixel based on the intensity of its neighbours.

The hamming distance represents the number of different bits in an encoding (census transform in this case).

## The overall algorithm:

- Apply the census transform on the greyscale input image
- Compute hamming costs given the census transform
- Optionally compute hamming sums for better results
- Compute disparity maps given the costs
