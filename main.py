import torch
import pandas as pd

from data.english_pos_tagging.load_test_data import load_data
from settings import Settings
from sequence_labeller import SequenceLabeller
from dataset import RawDataset
from metrics import get_metrics
from sklearn.model_selection import train_test_split


def pos_tagging_test():
    data = load_data()
    x_train, y_train = data['train_source'], data['train_target']
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train)

    train_data = RawDataset(sources=x_train, targets=y_train, features=None)
    dev_data = RawDataset(sources=x_dev[:10], targets=y_dev[:10], features=None)

    settings = Settings(
        name="pos_test", save_path="saved_models/test", loss="ctc-crf", device=torch.device("cuda:0"),
        report_progress_every=100, epochs=25, tau=2
    )

    labeller = SequenceLabeller(settings=settings)
    labeller = labeller.fit(train_data=train_data, development_data=None)
    # labeller = SequenceLabeller.load("saved_models/test/pos_test.pt")

    predictions = labeller.predict(data["test_source"][:10])
    print(predictions[0])

    predictions = [prediction.prediction for prediction in predictions]
    metrics = get_metrics(predictions=predictions, targets=data["test_target"][:10])
    print(metrics)


def inflection_test():
    train_data = pd.read_csv(
        "data/english_old/ang_large.train", sep="\t", header=None, names=["lemma", "form", "features"]
    )
    x_train = [["<PREFIX>"] + list(lemma) + ["<SUFFIX>"] for lemma in train_data["lemma"].tolist()]
    y_train = [list(form) for form in train_data["form"].tolist()]
    train_features = [feats.split(";") for feats in train_data["features"].tolist()]
    train_data = RawDataset(sources=x_train, targets=y_train, features=train_features)

    dev_data = pd.read_csv("data/english_old/ang.dev", sep="\t", header=None, names=["lemma", "form", "features"])
    x_dev = [["<PREFIX>"] + list(lemma) + ["<SUFFIX>"] for lemma in dev_data["lemma"].tolist()]
    y_dev = [list(form) for form in dev_data["form"].tolist()]
    dev_features = [feats.split(";") for feats in dev_data["features"].tolist()]
    dev_data = RawDataset(sources=x_dev, targets=y_dev, features=dev_features)

    settings = Settings(
        name="inflection_test", save_path="saved_models/test", loss="ctc", device=torch.device("cuda:0"),
        report_progress_every=100, epochs=50, tau=4, use_features=True, feature_pooling="mlp",
        hidden_size=256, num_layers=2, feature_num_layers=1, feature_hidden_size=128, gamma=0.9, batch_size=8
    )

    labeller = SequenceLabeller(settings=settings)
    labeller = labeller.fit(train_data=train_data, development_data=dev_data)
    # labeller = SequenceLabeller.load("saved_models/test/pos_test.pt")

    predictions = labeller.predict(sources=x_dev, features=dev_features)
    print(x_dev[0])
    print(y_dev[0])
    print(predictions[0])

    predictions = [prediction.prediction for prediction in predictions]
    metrics = get_metrics(predictions=predictions, targets=y_dev)
    print(metrics)


if __name__ == '__main__':
    inflection_test()
