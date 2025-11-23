Fruits & Vegetables Image Classification using CNN (TensorFlow & Keras)

This project builds a Convolutional Neural Network (CNN) model to classify 36 categories of fruits and vegetables using TensorFlow/Keras.
The model is trained on a structured dataset with train, validation, and test directories and achieves 95–96% validation accuracy.

📌 Project Overview

Built a deep learning classification model using TensorFlow, Keras, and CNN architectures.

Dataset contains 3115 training images, 351 validation images, and 359 test images across 36 classes.

The model predicts a fruit/vegetable image with high confidence using softmax outputs.

Achieved over 96% training accuracy and 95% validation accuracy.

📂 Dataset Structure
Fruits_Vegetables/
│── train/          (3115 images)
│── validation/     (351 images)
│── test/           (359 images)


Dataset contains 36 categories, including:
apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip, watermelon.

🧠 Model Architecture

A sequential CNN model with:

Rescaling layer

Conv2D + MaxPool layers (16, 32, 64 filters)

Flatten layer

Dropout (0.2)

Dense layer (128 units)

Output layer = 36 classes

Loss: SparseCategoricalCrossentropy
Optimizer: Adam

🚀 Training Performance

Training Epochs: 25

Final Training Accuracy: ~98%

Validation Accuracy: ~95–96%

Loss Stabilized around 0.25–0.40

Graphs included for:

Accuracy vs Epoch

Loss vs Epoch

🔍 Model Prediction
##**Fruits & Vegetables Image Classification using CNN (TensorFlow & Keras)**
Model predicts fruits/vegetables from a given image:

Veg/fruit in image is banana with an accuracy of 99.73%

💾 Saving the Model

Model is saved as:

Image_classify.keras

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

CNN for deep learning

Jupyter Notebook

📝 Resume Project Description (Point-Wise, 3–5 Points)

Built a CNN-based deep learning model using TensorFlow/Keras to classify 36 fruit and vegetable categories with 95%+ validation accuracy.

Preprocessed and structured dataset into train/validation/test splits and applied image augmentation & rescaling techniques.

Designed and trained a multi-layer CNN with Conv2D, MaxPooling, Dropout, and Dense layers for accurate classification.

Visualized model training performance using accuracy/loss curves and implemented softmax-based prediction for real images.

Exported and saved the trained model (Image_classify.keras) for deployment and future inference.
