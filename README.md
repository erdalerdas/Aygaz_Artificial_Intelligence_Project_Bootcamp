# [Accelerating Urban Analysis with CNN and Satellite Imagery]

## About the Project
  This project aims to develop a Convolutional Neural Network (CNN) model trained with satellite imagery to optimize the site analysis process of urban planners and landscape architects.

## Table of Contents
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Functions](#functions)
- [Usage](#usage)
  
## Requirements
To run the code in this project, you need the following libraries:

- pandas
- numpy
- os
- random
- tensorflow
- matplotlib
- cv2

You can install the required libraries using pip:

```bash
pip install pandas numpy os random tensorflow matplotlib cv2
```
## Project Structure
The main script in this project is `Aerial_Image_Classification.ipynb`, which includes all the functions and the execution code. Here is an overview of the key functions defined in the script.

## Functions

### 1. Data Loading and Preparation

*   The `data_load` function loads class names and creates a dictionary mapping class names to labels.
*   The `data_prep` function prepares data generators for training and validation.
*   The `split_data` function splits data into features (X) and labels (y).
*   The `visualize_images_with_labels` function visualizes random images with their labels.
  
### 2. Model Building

*   The `build_model` function builds a CNN model based on the input shape and number of classes.
  
### 3. Model Training

*   The `train_model` function trains the model with early stopping and returns training history.

### 4. Performance Visualization

*   The `plot_accuracy` function plots training and validation accuracy over epochs.
*   The `plot_loss` function plots training and validation loss over epochs.
  
### 5. Random Image Prediction

*   The `predict_random_image` function predicts the class and probabilities for a random image.

### Main Function

*   The `main` function runs all code sections and performs model training and evaluation.

## Usage
To use the script, follow these steps:
1. **Load the dataset**: Download and load the dataset from a [kaggle](https://www.kaggle.com/datasets/yessicatuteja/skycity-the-city-landscape-dataset/data).
3. Change the path inside the `main` function (data_directory = "path/to/data").
   
   <details>
     <summary>Details</summary>
     
     ```python
   
   
   def main():
    data_directory = "path/to/data"

    # Load and prepare data
    classes, label_classes = data_load(data_directory)
    train_generator, validation_generator = data_prep(data_directory, classes)

    # Split data into training and validation sets
    X_train, y_train = split_data(train_generator)
    X_test, y_test = split_data(validation_generator)

    # Visualize some images with labels
    visualize_images_with_labels(data_directory, num_images=20, font_size=20)

    # Build and train the model
    model = build_model(input_shape=(128, 128, 3), num_classes=len(classes))
    model.summary()
    model, history = train_model_with_best_accuracy(model, X_train, y_train, X_test, y_test, epochs=50, checkpoint_path='best_model.keras')

    # Plot accuracy and loss
    plot_accuracy(history)
    plot_loss(history)

    # Predict a random image from the dataset
    random_class, random_image, predicted_class, predictions = predict_random_image(model, data_directory, classes)
    print(f"Randomly selected image from class '{random_class}': {random_image}")
    print("Predicted class:", predicted_class)
    print("Class probabilities:", predictions)

    get_best_validation_metrics(history)
    ```
   </details>
5. Run the `main` function.
   
   ```python
    main()
   ```

