# Brain-Tumour-MRI-Image-Classification
# Brain Tumour MRI Classification

## 📚 **Project Overview**
This project focuses on the classification of brain MRI images into four categories:
- **No Tumor**
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

Using Convolutional Neural Networks (CNNs) and transfer learning architectures such as **ResNet50** and **VGG19**, the project aims to provide accurate predictions for MRI scans.

## 🛠️ **Technologies Used**
- **Python**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **Pandas**
- **NumPy**

## 📂 **Dataset**
- The dataset contains MRI images categorized into four classes.
- Images are sourced from a ZIP file and extracted into a structured dataset.
- **Classes:**
  - `no_tumor`
  - `glioma_tumor`
  - `meningioma_tumor`
  - `pituitary_tumor`

## 📊 **Data Preprocessing**
- Images are resized to **256x256 pixels**.
- Normalized pixel values between **0 and 1**.
- Data augmentation is applied to improve model performance.
- Dataset split:
  - **Training (60%)**
  - **Testing (40%)**
  - Further divided into training, validation, and test subsets.

## 🧠 **Model Architectures**
1. **Custom CNN Model:** A multi-layer convolutional neural network with dropout and batch normalization.
2. **Hyperparameter Tuned CNN:** Optimized CNN with adjusted dropout rates and batch normalization.
3. **ResNet50:** Transfer learning model pre-trained on ImageNet.
4. **VGG19:** Another transfer learning model with pre-trained weights.

## 🏋️‍♀️ **Training and Evaluation**
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam with tuned learning rates.
- **Metrics:**
  - Accuracy
  - Confusion Matrix
  - Classification Report

## 📈 **Results Visualization**
- Training and validation accuracy/loss graphs.
- Confusion matrices for training and test datasets.
- Classification reports with precision, recall, and F1 scores.

## 🔄 **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone <repo-link>
   ```
2. Upload the dataset ZIP file to Google Drive.
3. Update the `zip_path` variable in the script with the dataset path.
4. Run the notebook in Google Colab.
5. Monitor training and evaluation outputs.

## 📌 **Key Results**
- The CNN, ResNet50, and VGG19 models achieved high accuracy on the test dataset.
- Confusion matrices and classification reports showcase the performance.

## 🚀 **Inference on Custom Images**
- Provide a sample MRI image.
- The model predicts the tumor category with confidence scores.
- Results are displayed with the MRI image.

## 📑 **Future Improvements**
- Integrate **EfficientNet** for better accuracy.
- Experiment with advanced image augmentation techniques.
- Deploy the model using **Flask** or **Streamlit** for real-world use.

## 📜 **License**
This project is licensed under the **MIT License**.


