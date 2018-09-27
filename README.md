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

## module

### [skimage.io](http://scikit-image.org/docs/stable/api/skimage.io.html)
* ``io.imread(URL or xxx.jpg)`` : Load an image from file.
* ``io.imshow(img)`` : Display image
* ``io.imsave(NAME)`` : Save an image to file.
* ``io.ImageCollection`` : Load a collection of images.

### [skimage.color](http://scikit-image.org/docs/stable/api/skimage.color.html)
Do the color space conversion such as:
    1. RGB to HSV color space conversion. (``rgb2hsv(img)``)
    2. RGB to XYZ color space conversion. (``rgb2xyz(img)``)
    3. RGB to RGB CIE color space conversion. (``rgb2rgbcie(img)``)
    4. CIE-LAB to CIE-LCH color space conversion. (``lab2lch(img)``)
    
You can also use ``convert_colorspace(img, fromspace, tospace)`` to convert color space.

### [skimage.exposure](http://scikit-image.org/docs/stable/api/skimage.exposure.html)
* Histogram equalization : Make the intensity distribution better of image. It can get higher contrast.
	* [Histogram equalizatio ---wiki](https://en.wikipedia.org/wiki/Histogram_equalization)
	* ``exposure.histogram``  : histogram of image
	* ``exposure.equalize_hist`` : image after histogram equalization
	
* Gamma Correction : In human's vision, the brighteness to human is not in nonlinear. We should use gamma correction to make the brightness much linear in human sight.
	* [Gamma Correction ---wiki](https://en.wikipedia.org/wiki/Gamma_correction)
	* ``exposure.adjust_gamma(img, gamma_value)`` : Adjust gamma_value for image
	* ``exposure.is_low_contrast(img)`` : To justify whether the image is in low contrast

### skimage.data_dir
* data_dir : It is sample data directory under the absolute path where you setup scikit-image