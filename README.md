# IMDB Movie Review Sentiment Analysis

An end-to-end Deep Learning project that classifies IMDB movie reviews into **Positive** or **Negative** sentiments. The project utilizes a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras and is deployed as an interactive web application using Streamlit.

---

## Features

* **Deep Learning Model:** Simple RNN architecture with an Embedding layer trained on the IMDB Movie Reviews dataset.
* **Text Preprocessing Pipeline:** Automatic tokenization, lowercase conversion, word-index mapping, and sequence padding (`maxlen = 500`).
* **Interactive Web Interface:** Built with Streamlit for real-time sentiment prediction.
* **Confidence Score Output:** Displays both the predicted sentiment and the model's confidence score.
* **Pre-trained Model:** Includes saved model weights for quick deployment and inference.

---

## Model Architecture

The sentiment classifier is built using TensorFlow/Keras with the following architecture:

### 1. Input Layer

* Accepts tokenized and padded review sequences.

### 2. Embedding Layer

* Vocabulary Size: **10,000**
* Embedding Dimension: **128**
* Learns dense vector representations of words.

### 3. SimpleRNN Layer

* Hidden Units: **128**
* Activation Function: **ReLU**
* Captures sequential patterns and contextual information from reviews.

### 4. Output Layer

* Dense Layer with **1 neuron**
* Activation Function: **Sigmoid**
* Produces a probability score for binary classification.

### 5. Training Optimization

* **EarlyStopping Callback**

  * Monitors validation loss
  * Prevents overfitting
  * Restores best-performing model weights

---

## 📁 Repository Structure

```text
├── imdb_sentiment_analysis.ipynb   # Google Colab notebook for model training
├── prediction.ipynb                # Model loading and inference testing
├── main.py                         # Streamlit application
├── imdb-sentiment-analysis.h5      # Trained RNN model
└── README.md                       # Project documentation
```

---

## 💻 Tech Stack

| Category                | Technologies                   |
| ----------------------- | ------------------------------ |
| Language                | Python                         |
| Deep Learning           | TensorFlow, Keras              |
| Web Framework           | Streamlit                      |
| Numerical Computing     | NumPy                          |
| Development Environment | Google Colab, Jupyter Notebook |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vyom10445/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Create a Virtual Environment (Optional)

#### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy streamlit
```

Or:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run main.py
```



## 📊 Example

### Input

```text
This movie was absolutely fantastic. The acting, story, and cinematography were outstanding.
```

### Output

```text
Sentiment: Positive
Confidence Score: 0.96
```

---

## 👨‍💻 Author

**Vyom Chaturvedi**
