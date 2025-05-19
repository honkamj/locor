# Locor

Locor is a generic multimodal image registration tool based on modaling local functional dependence with linear combination of learned basis functions.

## Installation

First, install PyTorch with instructions at https://pytorch.org/ (GPU version highly recommended).

Then you can install Locor using pip by running the command

    pip install git+https://github.com/honkamj/locor

## Usage

The registration is invoked via running `python -m locor`. The most basic use case is:

    python -m locor <path_to_the_reference_image> <path_to_the_moving_image> -o <path_to_the_registered_moving_image.nii>

The images should be readable by nibabel (https://nipy.org/nibabel/). To see all the available options, run `python -m locor -h`.

Custom config files can be given with flag "-c". For Locor config files are defined as Python scripts defining a method "build_config". See the [default config](src/locor/default_config.py) for an example.

## Publication

If you use the method, please cite (see [bibtex](citations.bib)):

- **New multimodal similarity measure for image registration via modeling local functional dependence with linear combination of learned basis functions**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
Under review ([eprint arXiv:2503.05335](https://arxiv.org/abs/2503.05335 "eprint arXiv:2503.05335"))

## License

Locor tool is released under the MIT license.
