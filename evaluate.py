def evaluatef(detected_times, true_times, tolerance=2.0):
    matched_true = set()
    tp = 0

    for detected in detected_times:
        for i, true in enumerate(true_times):
            if i not in matched_true and abs(detected - true) <= tolerance:
                matched_true.add(i)
                tp += 1
                break

    fp = len(detected_times) - tp
    fn = len(true_times) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return precision, recall, f1