# Brain Tumour MRI Classification

##  **Project Overview**
This project focuses on the classification of brain MRI images into four categories:
- **No Tumor**
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

Using Convolutional Neural Networks (CNNs) and transfer learning architectures such as **ResNet50** and **VGG19**, the project aims to provide accurate predictions for MRI scans.

##  **Technologies Used**
- **Python**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **Pandas**
- **NumPy**
- **Google Colab**

## **Dataset**
- The dataset contains MRI images categorized into four classes.
- Images are sourced from a ZIP file and extracted into a structured dataset.
- **Classes:**
  - `no_tumor`
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`
- Images are split into **training**, **testing**, and **validation** sets.
- The dataset structure follows the pattern:
  ```
  Brain_tumor_dataset/
    ├── training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   ├── pituitary_tumor/
    ├── testing/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   ├── pituitary_tumor/
  ```

##  **Data Preprocessing**
- Images are resized to **256x256 pixels**.
- Normalized pixel values between **0 and 1**.
- Applied **Data Augmentation:**
  - Horizontal Flip
  - Vertical Flip
  - Rotation
  - Zoom
- Dataset split:
  - **Training (60%)**
  - **Validation (20%)**
  - **Testing (20%)**

##  **Model Architectures**
### 1. **Custom CNN Model:**
- 4 Convolutional layers
- MaxPooling layers
- Dropout layers to prevent overfitting
- Fully connected Dense layers

### 2. **Hyperparameter Tuned CNN:**
- Optimized dropout rates
- Batch normalization
- Tuned learning rate (0.0005)

### 3. **ResNet50:**
- Pre-trained on **ImageNet**
- Feature extraction with frozen convolutional layers
- Fully connected Dense layers

### 4. **VGG19:**
- Pre-trained on **ImageNet**
- Fully connected Dense layers for classification

##  **Training and Evaluation**
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (Learning Rates: 0.001, 0.0005, 0.0001)
- **Metrics:**
  - Accuracy
  - Validation Accuracy
  - Confusion Matrix
  - Classification Report
- Models trained with **early stopping** and **learning rate scheduling**.

##  **Results Visualization**
- Training and Validation **Accuracy/Loss Graphs**.
- **Confusion Matrices** for training and test datasets.
- **Classification Reports** with:
  - Precision
  - Recall
  - F1 Score

##  **Inference on Custom Images**
1. Provide a sample MRI image.
2. The image is preprocessed (resized and normalized).
3. The model predicts the tumor category with **confidence scores**.
4. The result is displayed with the MRI image and prediction details.

##  **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone <repo-link>
   ```
2. Upload the dataset ZIP file to **Google Drive**.
3. Update the `zip_path` variable in the script.
4. Run the script in **Google Colab**.
5. Monitor training and evaluation outputs.
6. Perform inference on custom images.

##  **Key Results**
- **Custom CNN Model:** Achieved consistent accuracy across training and testing datasets.
- **ResNet50 Model:** Effective for feature extraction with pre-trained weights.
- **VGG19 Model:** Balanced accuracy and computational efficiency.
- **Evaluation Metrics:** High precision, recall, and F1 scores across all classes.

##  **Future Improvements**
- **Integrate EfficientNet** for better accuracy.
- **Enhance Data Augmentation Techniques**.
- **Real-time Predictions:** Integrate with a real-time MRI scan system.
  
##  **License**
This project is licensed under the **MIT License**.

