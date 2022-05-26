def exponential_moving_avg(old_score: float, new_score: float, gamma: float = 0.95):
    if old_score is None:
        return new_score
    else:
        return gamma * old_score + (1 - gamma) * new_score
