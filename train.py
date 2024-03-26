import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def createModel():
    # Load data
    data = pd.read_csv('combined_data.csv')
    
    # Split into features and labels
    X = data['text']
    y = data['label']
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    
    # Pad sequences
    maxlen = 100  # Define maximum sequence length
    X_padded = pad_sequences(X_seq, maxlen=maxlen)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
    
    model.save("spamModel")
    return model