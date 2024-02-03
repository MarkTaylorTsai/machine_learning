# Prediction Script
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import re

def load_model_and_tokenizer():
    model = load_model('trained_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
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

def main():
    model, tokenizer = load_model_and_tokenizer()

    # Example new review or take user input
    test_review = input("Enter a review to analyze: ")
    sentiment = predict_sentiment(model, test_review, tokenizer)
    print(f"Review: {test_review}\nPredicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()