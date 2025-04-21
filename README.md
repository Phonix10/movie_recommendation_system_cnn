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
