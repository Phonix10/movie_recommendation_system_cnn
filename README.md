# 🎬 Movie Recommendation System using CNN

This project implements a *movie recommendation system* using *Convolutional Neural Networks (CNNs)* and content-based filtering. It utilizes movie poster images and genre metadata to recommend visually and contextually similar movies.

## 🧠 Features

- Content-based recommendations using visual features
- Trained CNN model to extract image embeddings
- Cosine similarity-based recommendation engine
- Simple UI built with Streamlit
- Content-based movie recommendation using CNN
- Built using TensorFlow and Keras

## 📁 Project Structure

```
.idea/
└── inspectionProfiles/
    └── profiles_settings.xml
.gitignore
misc.xml
modules.xml
movie_recommendation_system_cnn.iml
model/
    ├── emotion-detection.ipynb
    ├── model.json
    └── new_model.h5
.gitattributes
haarcascade_frontalface_default.xml
main.py
mode.ipynb
model_new.h5
model_weights.h5
model.json
README.md
testing.py
```

# 😊 Bonus Project: Facial Expression Recognition using CNN

In addition to the movie recommendation system, this repository includes a CNN-based **Facial Expression Recognition** model trained to classify human emotions such as **happy**, **angry**, **sad**, etc., using grayscale facial images.

## 🔍 Overview

- Input image size: 48x48 pixels (grayscale)
- Dataset: Images categorized by facial expressions (`happy`, `angry`, etc.)
- Data pipeline using `ImageDataGenerator`
- Deep CNN architecture with 5 convolutional layers
- Trained for 45 epochs with categorical cross-entropy loss
- Model saved as JSON (`model.json`) and weights as HDF5 (`new_model.h5`)

## 🧠 Model Architecture

- **Input Shape**: 48x48x1 (grayscale images)
- **Conv Layers**: 5 convolutional layers each followed by BatchNormalization, ReLU activation, MaxPooling, and Dropout
- **Dense Layers**:
  - Fully Connected Layer 1: 256 neurons
  - Fully Connected Layer 2: 512 neurons
  - Output Layer: 7 neurons with softmax activation (for 7 emotion classes)

## 🏗️ Training Procedure

```python
# Load and preprocess image data
ImageDataGenerator().flow_from_directory()

# Build CNN model
Sequential([
    Conv2D(...) → BatchNorm → ReLU → MaxPooling → Dropout,
    ...
    Flatten(),
    Dense(...) → BatchNorm → ReLU → Dropout,
    Dense(...) → BatchNorm → ReLU → Dropout,
    Dense(7, activation='softmax')
])


# Compile and train
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(...)
```

## ⚙ How It Works

1. *Train a CNN* to learn visual features from movie poster images.
2. *Extract feature vectors* and save them using numpy.
3. *Calculate cosine similarity* between feature vectors to find similar movies.
4. *Streamlit app* allows users to interact and view recommended movies.

## 🧪 Requirements
- Python 3.7+
- TensorFlow
- NumPy
- pandas
- scikit-learn
- matplotlib
- Streamlit

## 📚 Dataset
### The dataset includes:
- MovieGenre.csv: Movie titles and genre information
- Movie poster images in the images/ directory
- These are used to train the CNN and build the recommendation engine.

## 📈 Future Improvements
- Incorporate NLP on movie descriptions
- Combine content-based and collaborative filtering
- Enhance UI with better interactivity and visuals

## 👨‍💻 Author
Developed by Udit Ranjan And Prabhav Rathi.

## 🙌 Credits
This project was developed for learning and experimentation with deep learning in the domain of content-based recommendations.
