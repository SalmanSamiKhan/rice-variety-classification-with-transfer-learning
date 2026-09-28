# Rice variety classification with transfer learning

This project classifies five rice varieties from grain images:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

The notebook uses a MobileNetV2 model pretrained on ImageNet. The convolutional base stays frozen while a small classification head learns the rice classes. Training images receive light geometric augmentation. Validation and test images use the MobileNetV2 preprocessing without augmentation.

## Notebook

[`01_rice_variety_classification.ipynb`](notebooks/01_rice_variety_classification.ipynb)

The notebook covers dataset inspection, stratified splitting, image augmentation, transfer learning, training callbacks, test-set evaluation, a confusion matrix, and single-image prediction.

## Dataset

The notebook uses the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset), which contains 15,000 images for each of the five classes. Raw images are not included in this repository.

For local use, place the extracted data at:

```text
data/Rice_Image_Dataset/
├── Arborio/
├── Basmati/
├── Ipsala/
├── Jasmine/
└── Karacadag/
```

On Kaggle, attach the dataset to the notebook. The notebook checks both common Kaggle paths and the local path above.

## Reproduction

Install TensorFlow, NumPy, pandas, scikit-learn, matplotlib, and seaborn. Run the notebook from the first cell. A GPU is recommended because the dataset contains 75,000 images.

The notebook saves the trained model as `rice_classifier.keras` in the active working directory.

## Limitations

The images come from a single labelled dataset, so the reported test result measures performance on held-out images from the same source. It does not establish how well the classifier will work on photographs taken with different cameras, lighting, backgrounds, or grain preparation.
