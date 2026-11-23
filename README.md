# 🎬 IMDb Movie Review Sentiment Analyzer using Word2Vec and ANN

A Movie Review Sentiment Analysis project that classifies IMDb movie reviews as Positive or Negative using Word2Vec word embeddings and an Artificial Neural Network (ANN). The trained model is deployed as an interactive Streamlit web app.

## 🚀 Live Demo

🔗 **Try the Application:**
https://imdb-sentiment-analyzer-21t.streamlit.app/

## 📌 Project Overview

Sentiment Analysis is an important Natural Language Processing (NLP) task used to understand the opinions expressed in text.

In this project, I use:

- **Word2Vec** to convert words into numerical vectors.
- **Artificial Neural Network (ANN)** to classify movie reviews.
- **Streamlit** to build the web application.

The model predicts whether a given movie review expresses positive or negative sentiment.

## 🎯 Project Objectives

- Build a movie review sentiment classification system.
- Apply NLP preprocessing techniques to text data.
- Generate word embeddings using Word2Vec.
- Train an ANN model for sentiment classification.
- Deploy the model using Streamlit.
- Provide real-time predictions through a simple interface.

## 🧠 Model Architecture

### 1. Natural Language Processing

Text preprocessing includes:

- Lowercase conversion
- Contraction expansion (can't → cannot, didn't → did not)
- HTML tag and URL removal
- Special character, punctuation and digit removal
- Negation handling (not bad → not_bad)
- Tokenization

### 2. Word2Vec

Word2Vec converts words into dense numerical vectors that capture the meaning of words.

- Embedding Dimension: 100
- Algorithm: Skip-gram
- Window Size: 5

Each review is represented by averaging the vectors of its words.

### 3. Artificial Neural Network (ANN)

| Layer | Configuration |
|-------|---------------|
| Input Layer | 100 Features |
| Dense Layer 1 | 128 Neurons, ReLU |
| Dropout | 0.3 |
| Dense Layer 2 | 64 Neurons, ReLU |
| Dropout | 0.3 |
| Output Layer | 1 Neuron, Sigmoid |

- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Early Stopping: monitors validation loss

## 🔄 Project Workflow

1. Load the IMDb movie review dataset.
2. Preprocess the review text.
3. Tokenize the reviews.
4. Generate Word2Vec embeddings.
5. Convert reviews into sentence vectors.
6. Split the dataset into training and testing sets (80/20).
7. Train the ANN model.
8. Evaluate the trained model.
9. Save the trained models.
10. Build the Streamlit application.
11. Deploy the application on Streamlit Community Cloud.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Operations |
| Gensim | Word2Vec Embeddings |
| TensorFlow / Keras | ANN Model |
| Scikit-learn | Train-Test Split and Evaluation |
| Matplotlib, Seaborn | Visualization |
| Streamlit | Web Application |
| Streamlit Community Cloud | Deployment |
| Git & GitHub | Version Control |

## 📂 Project Structure

```
imdb-sentiment-analyzer/
│
├── app.py
├── requirements.txt
├── word2vec_project.ipynb
│
├── models/
│   ├── word2vec.model
│   └── sentiment_ann.keras
│
└── README.md
```

## ⚙️ Installation and Setup

**Step 1: Clone the Repository**

```bash
git clone https://github.com/swamyk18/imdb-sentiment-analyzer.git
```

**Step 2: Navigate to Project Folder**

```bash
cd imdb-sentiment-analyzer
```

**Step 3: Create Virtual Environment**

```bash
python -m venv venv
```

Activate the environment:

Windows

```bash
venv\Scripts\activate
```

Mac / Linux

```bash
source venv/bin/activate
```

**Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Run the Streamlit Application**

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 💻 Application Features

- Text box to enter a movie review
- Real-time sentiment prediction
- Confidence score for the prediction
- Probability bar for positive sentiment
- Warning when none of the words are known to the model
- Clear error message if the model files cannot be loaded

## 📊 Prediction Output

The application provides:

- **Positive Review:** the model predicts positive sentiment.
- **Negative Review:** the model predicts negative sentiment.
- **Confidence Score:** how sure the model is about its prediction.
- **Probability:** the probability of positive sentiment.

## 📈 Results

Evaluated on a test set of 10,000 reviews.

**Test Accuracy: 87.86%**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.89 | 0.87 | 0.88 | 5000 |
| Positive | 0.87 | 0.89 | 0.88 | 5000 |
| Macro Avg | 0.88 | 0.88 | 0.88 | 10000 |

## 🌐 Deployment

This project is deployed using Streamlit Community Cloud with Python 3.12.

🔗 **Live Application:**
https://imdb-sentiment-analyzer-21t.streamlit.app/

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing (NLP)
- Text preprocessing and tokenization
- Word2Vec word embeddings
- Sentence vector generation
- Artificial Neural Networks
- Model training, evaluation and saving
- Building interactive apps with Streamlit
- Deploying a machine learning app and fixing dependency issues

## 🚀 Future Improvements

- Compare with TF-IDF + Logistic Regression
- Try pre-trained embeddings (GloVe, FastText)
- Use LSTM or GRU layers to understand word order
- Show the words that influenced each prediction

## 👨‍💻 Author

**Swamy Kummari**

Computer Science Graduate | Aspiring AI/ML Engineer | Python, Machine Learning, Deep Learning, NLP, LLMs, Generative AI, LangChain, SQL

🔗 GitHub: https://github.com/swamyk18

## ⭐ Support

If you find this project useful, please give the repository a ⭐ on GitHub!

Built with Python, NLP, Word2Vec, Artificial Neural Network, TensorFlow and Streamlit.
