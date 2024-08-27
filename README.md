# Sentiment Analysis of Movie Reviews

## Project Overview
This project involves the development of an advanced text classification model using Keras and Python to perform sentiment analysis on movie reviews. The primary objective is to accurately predict the sentiment (positive or negative) expressed in textual data by leveraging various natural language processing (NLP) techniques and neural network architectures.

## Data
The dataset used for this project consists of movie reviews labeled with sentiment categories. Each review is preprocessed and converted into a format suitable for modeling, including tokenization and word embedding techniques.

### Data Processing:
- **Binary Sentiment Encoding:** The sentiment labels are encoded as binary values (0 for negative, 1 for positive).
- **Text Cleanup:** The reviews undergo preprocessing steps such as removing stop words, punctuation, and applying lowercasing to standardize the text.

## Modeling Approach
The project explores various NLP techniques to transform textual data into numerical representations that can be fed into machine learning models. 

### Key Techniques:
- **Tokenization:** Converting text into sequences of tokens, which are then mapped to numerical values.
- **Word Embeddings:** Applying word embeddings to represent words in a vector space, capturing semantic relationships between words.

## Neural Network Architectures
Multiple neural network architectures were engineered and evaluated to find the most effective model for sentiment prediction:

1. **Multi-Layer Perceptron (MLP):** A basic feedforward neural network used as a baseline model.
2. **1D Convolutional Neural Network (CNN):** Captures local patterns in the text by applying convolutional layers to the input sequences.
3. **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) that is capable of capturing long-term dependencies in the text, making it effective for sentiment analysis.

## Results
The project resulted in the development of robust models capable of accurately predicting the sentiment of movie reviews. The models were compared based on their accuracy, precision, recall, and F1-score, with the LSTM architecture generally providing the best performance.
