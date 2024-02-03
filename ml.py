import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def main():
    data = load_data('train.ft.txt')
    
    # preprocess data
    data = preprocess_data(data)

    # train and evaluate the model
    model, tokenizer = train_and_evaluate_model(data)

    # give a input to test
    test_review = input("Give me a review: ")

    setiment = predict_sentiment(model, test_review, tokenizer)
    print(f"Review: {test_review}\nPredicted Sentiment: {setiment}")

def load_data(filepath):
    labels = []
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            label, review = line.split(" ", 1)
            label = 1 if label == '__label__2' else 0
            labels.append(label)
            reviews.append(review.strip())
    
    data = pd.DataFrame({'Label': labels, 'Review': reviews})

    if data.isnull().values.any():
        print("Warning: NaN values found in the data")                          
    return data

def preprocess_data(data):

    data['Preprocess_Review'] = data['Review'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()).strip())

    return data

def extract_features(data):
    # initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['Preprocess_Review'])
    sequence = tokenizer.texts_to_sequences(data['Preprocess_Review'])
    padded_sequences = pad_sequences(sequence, padding='post', maxlen=200)

    return padded_sequences, tokenizer

def train_and_evaluate_model(data):
    # Split the data 
    X, tokenizer = extract_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, data['Label'], test_size=0.2, random_state=42)

    # build the model
    model = Sequential([
        Embedding(10000, 16, input_length=200),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    adam = Adam(learning_rate=0.001)
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    
    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy: ", accuracy)
    print("Loss: ", loss)

    model.save('trained_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, tokenizer

def predict_sentiment(model, new_data, tokenizer):
    new_data_processed = re.sub(r'\W+', ' ', new_data.lower()).strip()

    # convert the new data to a sequence od integers using the tokenizer
    new_data_sequence = tokenizer.texts_to_sequences([new_data_processed])

    # pad the sequence to be the same length as the training data
    new_data_padded = pad_sequences(new_data_sequence, padding='post', maxlen=200)

    # predict the setiment using the trained model
    prediction = model.predict(new_data_padded)

    if prediction[0] >= 0.5:
        return "Positive Review"
    else:
        return "Negative Review"

if __name__ == "__main__":
    main()

