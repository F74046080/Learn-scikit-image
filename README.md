# Learn-scikit-image
This repo is to learn some image processing in python by scikit-image module.

## setup
### Windows for conda-based distributions
```bash
$ conda install scikit-image
```
### Linux and OSX
```bash
$ pip install -U scikit-image
```

## function
### skimage.color
Do the color space conversion such as:
    1. RGB to HSV color space conversion. (``rgb2hsv(img)``)
    2. RGB to XYZ color space conversion. (``rgb2xyz(img)``)
    3. RGB to RGB CIE color space conversion. (``rgb2rgbcie(img)``)
    4. CIE-LAB to CIE-LCH color space conversion. (``lab2lch(img)``)
    
You can also use ``convert_colorspace(img, fromspace, tospace)`` to convert color space.

### skimage.exposure
* ``exposure.histogram``  : histogram of image
* ``exposure.equalize_hist`` : image after histogram equalization

Histogram equalization : Make the intensity distribution better of image. It can get higher contrast.
[wiki](https://en.wikipedia.org/wiki/Histogram_equalization)
### skimage.data_dir