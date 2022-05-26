import os

from models import NeuralCRF
from encoders import LSTMEncoder
from metrics import token_accuracy
from models import NeuralSequenceLabeller
from experiments.english_pos_tagging.load_test_data import load_data


def pos_tagging_experiment():
    data = load_data()
    x_train, y_train = data['train_source'], data['train_target']

    model_class = NeuralSequenceLabeller

    # Train
    encoder = LSTMEncoder()
    model = model_class(encoder, epochs=1, save_path="experiments/english_pos_tagging/models/test")
    model.fit(x_train, y_train)

    model = model_class.load(os.path.join("experiments/english_pos_tagging/models/test", "labeller.pt"))

    # Evaluate
    x_test, y_test = data['test_source'], data['test_target']
    y_predicted = model.predict(x_test)

    print(f"Correct tags: {100 * token_accuracy(y_test, y_predicted):.2f}%")
