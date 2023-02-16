from preprocessing.tokenizer import Tokenizer
import os
import pickle
import re

class TextProcessor:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None

    def __load_tokenizer(self):
        if os.path.exists(self.tokenizer_path + "/tokenizer.pkl") == True:
            with open(self.tokenizer_path + "/tokenizer.pkl", 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            self.tokenizer = Tokenizer()
            if os.path.exists(self.tokenizer_path) == False:
                os.mkdir(self.tokenizer_path)
    
    def __save_tokenizer(self):
        if os.path.exists(self.tokenizer_path) == False:
            os.mkdir(self.tokenizer_path)
        with open(self.tokenizer_path + "/tokenizer.pkl", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __build_tokenizer(self, sequences: list):
        self.tokenizer.fit_to_texts(sequences)

    def process(self, sequences: list, max_length: int, padding: str = "post", truncating: str = "post"):
        # self.__load_tokenizer()
        digit_sequences = self.tokenizer.texts_to_sequences(sequences)
        padded_sequences = self.tokenizer.pad_sequences(digit_sequences, maxlen=max_length, padding=padding, truncating=truncating)

        return padded_sequences

    def fit(self, sequences: list):
        self.__load_tokenizer()
        self.__build_tokenizer(sequences=sequences)
        self.__save_tokenizer()

