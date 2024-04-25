# Terrain Inpainting using Unet and Partial Convolution

This is an implementation of the NVIDIA paper on partial convolution for inpainting specifically for 2D heightmaps.

## Author

- **Author Name**: You, Jisang
- **Email**: jisangyou1@gmail.com
- **Lab**: RAILAB, KAIST

## Installation

To run the code, you'll need to install the partial convolution layer. You can use pip3:

```bash
pip3 install torch_pconv
```

You'll also need to install these: torch, torchvision, pillow, and matplotlib

**Tested on python3.10

## References

- [Image Inpainting with Irregular Holes using Partial Convolutions](https://research.nvidia.com/publication/2018-09_image-inpainting-irregular-holes-using-partial-convolutions)

- [PyTorch Partial Convolution Inpainting](https://github.com/tanimutomo/partialconv) - Loss function and mask generator from here

## Heightmap Data Generation:

The heightmap data for training is generated using Raisim lidar located at:

/home/jakob/Desktop/ME491/raisimLib/examples/src/server
