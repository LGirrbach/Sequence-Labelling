import torch

from data.english_pos_tagging.load_test_data import load_data
from settings import Settings
from sequence_labeller import SequenceLabeller
from dataset import RawDataset
from metrics import get_metrics
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = load_data()
    x_train, y_train = data['train_source'], data['train_target']
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train)

    train_data = RawDataset(sources=x_train, targets=y_train)
    dev_data = RawDataset(sources=x_dev, targets=y_dev)

    settings = Settings(
        name="pos_test", save_path="saved_models/test", loss="cross-entropy", device=torch.device("cuda:0"),
        report_progress_every=100, epochs=20, tau=1
    )

    labeller = SequenceLabeller(settings=settings)
    labeller = labeller.fit(train_data=train_data, development_data=dev_data)

    predictions = labeller.predict(data["test_source"])
    print(predictions[0])

    predictions = [prediction.prediction for prediction in predictions]
    metrics = get_metrics(predictions=predictions, targets=data["test_target"])
    print(metrics)
