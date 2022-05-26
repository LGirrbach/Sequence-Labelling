def load_pos_data(file: str):
    source, target = [], []
    current_source, current_target = [], []

    with open(file) as df:
        for line in df:
            line = line.strip()

            if not line:
                if current_source and current_target:
                    source.append(current_source)
                    target.append(current_target)
                    current_source, current_target = [], []

                continue

            word, pos_tag = line.split(" ")
            current_source.append(word)
            current_target.append(pos_tag)

        if current_source and current_target:
            source.append(current_source)
            target.append(current_target)

    return source, target


def load_data():
    train_source, train_target = load_pos_data("experiments/english_pos_tagging/pos.train.txt")
    test_source, test_target = load_pos_data("experiments/english_pos_tagging/pos.test.txt")

    return {
        'train_source': train_source,
        'train_target': train_target,
        'test_source': test_source,
        'test_target': test_target
    }
