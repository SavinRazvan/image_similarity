# Image Similarity Comparison using VGG16

This project compares the similarity between images using features extracted from the VGG16 pre-trained deep learning model. It uses cosine similarity to compute the similarity scores between feature vectors.

## Project Structure

```bash
📦image_similarity
 ┣ 📂data
 ┃ ┗ 📂images
 ┃ ┃ ┣ 📜cat1.jpg
 ┃ ┃ ┣ 📜cat2.jpg
 ┃ ┃ ┣ 📜dog1.jpg
 ┃ ┃ ┗ 📜dog2.jpg
 ┣ 📜LICENSE.txt
 ┣ 📜README.md
 ┣ 📜image_similarity.ipynb
 ┗ 📜requirements.txt
```

## Requirements

To install the necessary libraries, use the following command:

```bash
pip install -r requirements.txt
```

### requirements.txt

```plaintext
opencv-python
numpy
matplotlib
scikit-image
scikit-learn
keras
tensorflow
```

## How to Run the Project

1. Clone the repository to your local machine.
2. Ensure you have Python installed.
3. Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4. Place your images in the `data/images/` directory.
5. Open and run the `image_similarity.ipynb` notebook to compare the images.

## image_similarity.ipynb

The `image_similarity.ipynb` notebook contains the following sections:

### Section 1: Import Libraries

**Description**: This cell imports all the necessary libraries for image processing, feature extraction, and similarity calculation. These include:
- `opencv-python` for image processing.
- `numpy` for numerical operations.
- `matplotlib` for plotting.
- `scikit-image` for additional image processing functions.
- `scikit-learn` for similarity calculations.
- `keras` and `tensorflow` for using the pre-trained VGG16 model and deep learning operations.

**Purpose**: To ensure that all required libraries are imported and available for use in subsequent cells.

### Section 2: Load and Preprocess Image Function

**Description**: This cell defines a function `load_and_preprocess_image` that:
- Loads an image from the specified path.
- Resizes the image to a target size (224x224) required for VGG16.
- Applies necessary preprocessing steps like scaling pixel values using `keras`'s `preprocess_input`.

**Purpose**: To handle the loading and preprocessing of images, preparing them for input into the VGG16 model.

### Section 3: Extract Features Using VGG16

**Description**: This cell defines a function `extract_vgg16_features` that:
- Loads the pre-trained VGG16 model with weights trained on ImageNet.
- Creates a new model that outputs features from the 'fc1' layer of VGG16.
- Extracts and flattens the features from the preprocessed image.

**Purpose**: To use the pre-trained VGG16 model to extract deep features from the preprocessed image, which are used for comparing the images.

### Section 4: Calculate Similarity Function

**Description**: This cell defines a function `calculate_similarity` that:
- Computes the cosine similarity between two given feature vectors using `scikit-learn`'s `cosine_similarity` function.

**Purpose**: To calculate the cosine similarity between two feature vectors, providing a measure of similarity that ranges between -1 and 1.

### Section 5: Display Images Function

**Description**: This cell defines a function `display_images` that:
- Displays a list of images along with their titles in a single figure using `matplotlib`.

**Purpose**: To visually display the images along with their titles, helping to verify the images being compared and understand the context of the similarity scores.

### Section 6: Plot Similarities Function

**Description**: This cell defines a function `plot_similarities` that:
- Creates a bar plot to visualize the pairwise similarity scores between the images using `matplotlib`.

**Purpose**: To provide an intuitive visual representation of the similarity scores, showing how similar each pair of images is based on the extracted features.

### Section 7: Compare Images Function

**Description**: This cell defines a function `compare_images` that:
- Loads and preprocesses each image.
- Extracts features from each image using the VGG16 model.
- Calculates pairwise similarities between the images.
- Displays the images.
- Plots the similarity scores.
- Prints the similarity results.

**Purpose**: To orchestrate the complete image comparison process by integrating all the previously defined functions, from loading and preprocessing images to displaying results.

### Section 8: Main Function

**Description**: This cell defines the `main` function that:
- Specifies the list of image paths to be compared.
- Calls the `compare_images` function to execute the comparison.

**Purpose**: To act as the entry point for the script, specifying the images to compare and initiating the comparison process.

### Section 9: Run the Main Function

**Description**: This cell runs the `main` function.

**Purpose**: To start the image comparison process when the notebook is executed.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.
