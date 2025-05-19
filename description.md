# Locor

Locor is a generic multimodal image registration tool based on modaling local functional dependence with learned basis functions.

## Installation

First, install PyTorch with instructions at https://pytorch.org/ (GPU version highly recommended).

Then you can install Locor using pip by running the command

    pip install git+https://github.com/honkamj/locor

## Usage

The registration is invoked via running `python -m locor`. The most basic use case is:

    python -m locor <path_to_the_reference_image> <path_to_the_moving_image> -o <path_to_the_registered_moving_image.nii>

The images should be readable by nibabel (https://nipy.org/nibabel/). To see all the available options, run `python -m locor -h`.

Custom config files can be given with flag "-c". For Locor config files are defined as Python scripts defining a method "build_config". See the [default config](https://github.com/honkamj/locor/blob/main/src/locor/default_config.py) for an example.

## Source code

Code repository can be found at GitHub [https://github.com/honkamj/locor](https://github.com/honkamj/locor).

## Publication

If you use deformation inversion layer, please cite the Locor paper. Details for citing can be found at the GitHub repository.

## License

Locor tool is released under the MIT license.
