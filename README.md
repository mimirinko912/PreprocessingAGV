# Agriculture-Vision 2021 NPY Preprocessor

**A highly efficient preprocessing toolkit for the Agriculture-Vision 2021 dataset.**

This repository provides a robust pipeline to convert the raw, fragmented Agriculture-Vision 2021 dataset (images, boundaries, and multi-class labels) into clean, unified `.npy` (NumPy) format, ready for deep learning training (PyTorch/TensorFlow).

## Key Features

- **Channel Fusion**: Merges RGB (3-channel) and NIR (1-channel) images into a single **4-channel** tensor (`H, W, 4`).
- **Boundary Handling**: Automatically applies valid-region masks (boundaries) to zero out invalid pixels in the source images.
- **Label Unification**: Aggregates separate label folders (e.g., Water, Weeds, Drydown) into a single semantic segmentation mask.
- **NPY Serialization**: Saves processed data as `.npy` files for significantly faster I/O during model training.
- **Modular Utils**: Code is organized into decoupled utility modules for easy maintenance and extension.

## Project Structure

We organize the code into logical modules to keep everything clean:

```text
AgriVision2021-Preprocessor/
├── data/                   # (Optional) Place your raw dataset here
├── output/                 # Generated .npy files will be saved here
├── src/
│   ├── __init__.py
│   ├── io_utils.py         # Handles file finding, path parsing, and saving NPY
│   ├── img_utils.py        # RGB+NIR merging, boundary masking logic
│   └── label_utils.py      # Multi-label merging and conflict resolution
├── main.py                 # Entry point script
├── requirements.txt        # Python dependencies
└── README.md
```
