# Autonomous Vehicle Lane and Edge Detection with UNet

## Project Overview

This project implements an **autonomous vehicle perception system** focused on **lane and edge detection** using deep learning and image processing techniques. The core model is a **UNet-based architecture**, which is a convolutional neural network (CNN) specialized for semantic segmentation tasks. Additionally, various image preprocessing techniques are used to enhance lane detection under different conditions (e.g., using HSV, HSL, and RGB color spaces). This project combines classical computer vision and deep learning techniques, leveraging **PyTorch**, **OpenCV**, and **image segmentation** to create a robust solution for real-time lane detection.

## Project Structure

- **main.py**: Orchestrates the overall pipeline, managing data loading, model training, and evaluation.
- **load_data.py**: Contains functions to load and preprocess image datasets, handling both training and validation sets.
- **train_model.py**: Implements the training and validation loop, including loss calculation and backpropagation for model optimization.
- **unet.py**: Defines the UNet model architecture used for lane and edge detection.
- **visual.py**: Provides visualization utilities for displaying predictions, losses, and segmentation outputs.
- **edge_detection.py**: Implements edge detection using traditional image processing methods (e.g., Canny edge detection).
- **lane_detection.py**: Utilizes image processing and filtering techniques to enhance lane features in images.
- **laneHSL.py, laneHSV.py, laneRGB.py**: Each file applies lane detection in different color spaces (HSL, HSV, and RGB), enabling flexibility across varying lighting and environmental conditions.

## Technical Details

### 1. Data Loading and Preprocessing (load_data.py)
Data loading involves reading image datasets and preprocessing them to match the input dimensions required by the UNet model. **Data augmentations** are applied to improve model generalization, such as random rotations, flips, and brightness adjustments, simulating real-world variations in road images.

### 2. UNet Model Architecture (unet.py)
The core of the lane detection system is the **UNet model**, a type of CNN well-suited for image segmentation. Key aspects include:
   - **Encoder-Decoder Structure**: The UNet model has a contracting path (encoder) and an expanding path (decoder), allowing it to capture context and precise localization.
   - **Skip Connections**: Skip connections between corresponding encoder and decoder layers preserve spatial information lost during downsampling.
   - **Activation and Final Output**: A softmax or sigmoid activation function at the output layer produces a pixel-wise classification map for lane and edge boundaries.

### 3. Model Training and Optimization (train_model.py)
The training loop manages **forward propagation**, **loss calculation**, **backpropagation**, and **parameter updates** for each epoch. Key components include:
   - **Loss Function**: A combination of **binary cross-entropy** and **Dice loss** optimizes for accurate pixel segmentation while managing class imbalance.
   - **Optimizer**: **Adam optimizer** is used for faster convergence, with parameters tuned for minimizing the loss function.
   - **Validation Loop**: After each epoch, the model is evaluated on a validation set, with metrics logged to monitor overfitting and ensure the model’s robustness.

### 4. Edge Detection (edge_detection.py)
This module applies traditional image processing for detecting image edges, crucial for enhancing lane boundaries:
   - **Canny Edge Detection**: Identifies strong gradients that often correspond to lane and road boundaries.
   - **Thresholding and Dilation**: Used to strengthen detected edges, allowing more reliable segmentation inputs for lane marking identification.

### 5. Lane Detection Using Color Spaces (laneHSL.py, laneHSV.py, laneRGB.py)
These files implement lane detection in different color spaces to enhance visibility under various lighting conditions:
   - **HSV Color Space**: Suitable for highlighting lanes under shadowed or high-contrast lighting conditions.
   - **HSL Color Space**: Effective in isolating lane colors, especially when there are changes in brightness.
   - **RGB Color Space**: Applied in scenarios where color differences alone suffice for lane marking detection.
   - **Gaussian and Median Filtering**: Each color space module applies these filters to reduce noise, followed by binary thresholding to isolate lanes.

### 6. Visualization and Results (visual.py)
Visualization functions help in analyzing model performance, including:
   - **Segmentation Maps**: Displays ground truth and predicted segmentation masks for lane boundaries.
   - **Training Metrics**: Plots losses and accuracy metrics across epochs, aiding in diagnosing overfitting or underfitting issues.
   - **Edge and Lane Detection Visualizations**: Illustrates the output of edge detection and lane detection filters, providing qualitative feedback on preprocessing effects.

### 7. Integration and Execution (main.py)
The `main.py` script combines data loading, model training, and evaluation, creating a complete pipeline for training and deploying the lane detection model:
   - **Command Line Arguments**: Provides flexibility to specify dataset paths, training parameters, and model save paths.
   - **Real-Time Processing**: Designed for batch processing but adaptable to real-time lane detection with pre-trained models.

## Skills and Knowledge Demonstrated

- **Deep Learning and Model Design**: Designed and trained a UNet-based model, demonstrating proficiency in CNN architectures for segmentation.
- **Image Processing**: Applied advanced techniques like Canny edge detection, Gaussian filtering, and thresholding to enhance image quality and lane detection accuracy.
- **Color Space Manipulation**: Demonstrated understanding of HSV, HSL, and RGB color spaces to handle lighting variations in real-world images.
- **Data Augmentation**: Improved model generalization by applying data augmentation techniques, addressing domain shifts commonly seen in autonomous driving data.
- **Loss Functions and Optimization**: Used combined loss functions to address class imbalance and optimized model parameters using Adam.
- **Model Evaluation and Visualization**: Utilized various visualization techniques to assess model performance and interpret segmentation outputs effectively.
- **Technical Documentation and Reporting**: Created detailed code and project documentation, illustrating key technical decisions and results.

## Reflections and Acknowledgements

This project combines deep learning and classical image processing techniques to address lane detection challenges in autonomous driving. The project required a deep understanding of computer vision, segmentation, and neural network design, highlighting practical skills in **image processing** and **deep learning**. Resources such as **PyTorch tutorials**, **OpenCV documentation**, and contributions from peers were invaluable.

