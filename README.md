# image_processor
DESCRIPTION
This is a package that can be used to create a custom image dataset.
It uses Pytorch functionality to,
1. Load image data
2. Load image labels
3. Apply custom transforms to the images and finally converts them into tensors for deep learning.
4. Creates a dataset object from which a DataLoader object could be created.
5. Can also be used to denormalize and view images using the "viewImages" function.

INSTALLATION
run "pip install DatasetCreator" in your python interpretor

USAGE
from DatasetCreator import PlantDataset, viewImages
PlantDataset()
