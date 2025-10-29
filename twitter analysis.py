# Import necessary libraries
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, SimpleRNN, Embedding, LSTM, Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk

# Load training and validation datasets
df = pd.read_csv('twitter_training.csv')
df1 = pd.read_csv('twitter_validation.csv')

# Data preprocessing for training data
df = df.dropna(axis=0)  # Remove rows with missing values

# Extract features (tweet text) - note: column name appears to be actual tweet text
x = df['im getting on borderlands and i will murder you all ,']

# Convert to list for processing
a = []
for i in x:
    a.append(i)

# Extract target labels
y = df['Positive']

# Encode categorical labels to numerical values
le = LabelEncoder()
y = le.fit_transform(y)
y_train = y

# Text preprocessing: remove punctuation
for ch in ['?', ',', '!', '.']:
    for i in a:
        i = i.replace(ch, "")

# Tokenize text data
tkn = Tokenizer()
tkn.fit_on_texts(a)  # Build vocabulary
sequences = tkn.texts_to_sequences(a)  # Convert text to sequences
padded = pad_sequences(sequences, maxlen=250)  # Pad sequences to uniform length
x_train = padded

# Load validation dataset
df1 = pd.read_csv('twitter_validation.csv')
df1.info()  # Display dataset information

# Process test data
x_test = df1['I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom\'s great auntie as \'Hayley can\'t get out of bed\' and told to his grandma, who now thinks I\'m a lazy, terrible person 🤣']

b = []
for i in x_test:
    b.append(i)

# Text preprocessing for test data
for ch in ['?', ',', '!', '.']:
    for i in b:
        i = i.replace(ch, "")

# Tokenize and pad test data using same tokenizer
tkn.fit_on_texts(b)
sequences = tkn.texts_to_sequences(b)
padded1 = pad_sequences(sequences, maxlen=250)
x_testt = padded1

# Process test labels
y_test = df1['Irrelevant']
y_test = np.array(y_test)
y_test = le.fit_transform(y_test)  # Note: Should use transform() not fit_transform() to maintain consistency

# Model architecture parameters
vocab_size = 40000    # Maximum number of unique words in vocabulary
embedding_dim = 128   # Dimension of word embeddings
maxlen = 250          # Maximum sequence length
num_classes = 4       # Number of output classes (sentiment categories)

# Build LSTM model for sentiment analysis
model = Sequential()

# Embedding layer: converts word indices to dense vectors
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))

# LSTM layer: processes sequences with memory, better for text than SimpleRNN
model.add(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))

# Fully connected layer with ReLU activation
model.add(Dense(64, activation='relu'))

# Dropout layer for regularization to prevent overfitting
model.add(Dropout(0.5))

# Output layer with sigmoid activation for multi-class classification
model.add(Dense(num_classes, activation='sigmoid'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
              optimizer='adam',                        # Adam optimizer for adaptive learning rates
              metrics=['accuracy'])                    # Track accuracy during training

# Display model architecture
model.summary()

# Train the model
h = model.fit(x_train, y_train, 
              epochs=5,                # Number of training iterations
              batch_size=64,           # Number of samples per gradient update
              validation_data=(x_testt, y_test))  # Use validation data to monitor performance