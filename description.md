# Locor

Locor is a generic multimodal image registration tool based on modaling local functional dependence with learned basis functions.

## Installation

Install using pip by running the commands

    pip install git+https://github.com/honkamj/locor
    pip install git+https://github.com/honkamj/composable-mapping@d7e28b68017f359840fdb77850e0dc5a9ddd9431

## Usage

The registration is invoked via running `python -m locor`. The most basic use case is:

    python -m locor <path_to_the_fixed_image> <path_to_the_moving_image> -o <path_to_the_registered_moving_image.nii>

The images should be readable by nibabel (https://nipy.org/nibabel/). To see all the available options, run `python -m locor -h`.

## Source code

Code repository can be found at GitHub [https://github.com/honkamj/locor](https://github.com/honkamj/locor).

## Publication

If you use deformation inversion layer, please cite the Locor paper. Details for citing can be found at the GitHub repository.

## License

Locor tool is released under the MIT license.
