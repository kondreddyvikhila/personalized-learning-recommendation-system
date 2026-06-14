# 🎓 Personalized Learning Recommendation System

## 🌐 Live Demo
👉 https://personalized-learning-recommendation-system-6zwtvqxj8uonyo2xug.streamlit.app/

---

## 📌 Project Overview

This project is an AI-powered Personalized Learning Recommendation System that suggests courses based on user interests, skills, and learning level.

It uses Machine Learning techniques (TF-IDF and Cosine Similarity) to recommend similar courses and generate a personalized learning path.

---

## 🚀 Features

- Course recommendation based on similarity
- Hybrid ranking (content + rating)
- Personalized learning path (Beginner → Advanced)
- Interactive Streamlit web app
- Real-time recommendations

---

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

---

## ⚙️ How It Works

1. User selects a course  
2. Features are combined (title + skills + level)  
3. TF-IDF converts text into vectors  
4. Cosine similarity finds similar courses  
5. Top recommendations are displayed  

---

## 📁 Project Structure

app.py  
data_preprocessing.py  
final_courses_dataset.csv  
requirements.txt  

---

## 🚀 How to Run Locally

pip install -r requirements.txt  
streamlit run app.py  

---

## 👨‍💻 Author

Vikhila Kondreddy  

---

## ⭐ Future Improvements

- Add user login system  
- Improve recommendation accuracy  
- Add feedback-based learning  
- Deploy with database integration  
