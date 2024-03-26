import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from train import createModel

def test_email_text(sample_email, model, maxlen, tokenizer):
    sample_email = sample_email.lower()
    # Tokenize and pad the sample email text
    sample_seq = tokenizer.texts_to_sequences([sample_email])
    sample_padded = pad_sequences(sample_seq, maxlen=maxlen)
    
    # Predict using the trained model
    prediction = model.predict(sample_padded)
    
    # Round the prediction to get the final label (0 for not spam, 1 for spam)
    predicted_label = np.round(prediction)[0][0]
    
    if predicted_label == 1:
        print("The sample email is predicted to be spam.")
    else:
        print("The sample email is predicted to be not spam.")
    
    print("Spam probability:", prediction[0][0])
    
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('combined_data.csv')

    # Extract text data to be tokenized
    X = data['text']

    # Prepare tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    maxlen = 100  # Define maximum sequence length
    
    # Check if model is stored
    model = None
    if os.path.exists("./spamModel"):
        model = tf.keras.models.load_model("spamModel")
    else:
        model = createModel()
    
    while True:
        test_email_text(str(input("Enter email text: ")), model, maxlen, tokenizer)