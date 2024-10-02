import os
import pickle


def load_model(name):
    with open(os.path.join(os.getcwd(), 'models', name), 'rb') as f:
        return pickle.loads(f.read())