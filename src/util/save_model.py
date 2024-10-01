import pickle


def save_model(model, path):
    with open(path, "wb+") as f:
        f.write(pickle.dumps(model))
