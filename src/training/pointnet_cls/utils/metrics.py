import numpy as np


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def binary_confusion_matrix(labels, preds):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    preds = np.asarray(preds, dtype=np.int32).reshape(-1)
    confusion = np.zeros((2, 2), dtype=np.int32)
    for true_label, pred_label in zip(labels, preds):
        confusion[int(true_label), int(pred_label)] += 1
    return confusion


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _class_metrics(labels, preds, class_index):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    preds = np.asarray(preds, dtype=np.int32).reshape(-1)
    true_positive = int(np.sum((labels == class_index) & (preds == class_index)))
    false_positive = int(np.sum((labels != class_index) & (preds == class_index)))
    false_negative = int(np.sum((labels == class_index) & (preds != class_index)))
    support = int(np.sum(labels == class_index))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def compute_eval_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    preds = np.asarray(preds, dtype=np.int32).reshape(-1)
    confusion = binary_confusion_matrix(labels, preds)
    tn, fp, fn, tp = confusion.ravel()

    complete_precision = _safe_divide(tn, tn + fn)
    complete_recall = _safe_divide(tn, tn + fp)
    complete_f1 = (
        2.0 * complete_precision * complete_recall / (complete_precision + complete_recall)
        if complete_precision + complete_recall
        else 0.0
    )

    broken_precision = _safe_divide(tp, tp + fp)
    broken_recall = _safe_divide(tp, tp + fn)
    broken_f1 = (
        2.0 * broken_precision * broken_recall / (broken_precision + broken_recall)
        if broken_precision + broken_recall
        else 0.0
    )

    return {
        "accuracy": float(np.mean(labels == preds)),
        "confusion_matrix": confusion,
        "complete_precision": complete_precision,
        "complete_recall": complete_recall,
        "complete_f1": complete_f1,
        "broken_precision": broken_precision,
        "broken_recall": broken_recall,
        "broken_f1": broken_f1,
        "macro_f1": 0.5 * (complete_f1 + broken_f1),
    }


def build_classification_report(labels, preds, label_names):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    preds = np.asarray(preds, dtype=np.int32).reshape(-1)

    rows = []
    for class_index, label_name in enumerate(label_names):
        class_metrics = _class_metrics(labels, preds, class_index)
        rows.append((label_name, class_metrics))

    accuracy = float(np.mean(labels == preds))
    macro_precision = float(np.mean([row["precision"] for _, row in rows]))
    macro_recall = float(np.mean([row["recall"] for _, row in rows]))
    macro_f1 = float(np.mean([row["f1"] for _, row in rows]))
    total_support = int(len(labels))
    weighted_precision = _safe_divide(
        sum(row["precision"] * row["support"] for _, row in rows), total_support
    )
    weighted_recall = _safe_divide(
        sum(row["recall"] * row["support"] for _, row in rows), total_support
    )
    weighted_f1 = _safe_divide(
        sum(row["f1"] * row["support"] for _, row in rows), total_support
    )

    lines = []
    lines.append("{:>16s} {:>10s} {:>10s} {:>10s} {:>10s}".format("", "precision", "recall", "f1-score", "support"))
    lines.append("")
    for label_name, row in rows:
        lines.append(
            "{:>16s} {:10.4f} {:10.4f} {:10.4f} {:10d}".format(
                label_name,
                row["precision"],
                row["recall"],
                row["f1"],
                row["support"],
            )
        )
    lines.append("")
    lines.append("{:>16s} {:>10s} {:>10s} {:10.4f} {:10d}".format("accuracy", "", "", accuracy, total_support))
    lines.append(
        "{:>16s} {:10.4f} {:10.4f} {:10.4f} {:10d}".format(
            "macro avg",
            macro_precision,
            macro_recall,
            macro_f1,
            total_support,
        )
    )
    lines.append(
        "{:>16s} {:10.4f} {:10.4f} {:10.4f} {:10d}".format(
            "weighted avg",
            weighted_precision,
            weighted_recall,
            weighted_f1,
            total_support,
        )
    )
    return "\n".join(lines)


def binary_roc_curve(labels, scores, positive_label=1):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    positives = labels == positive_label
    total_positive = int(np.sum(positives))
    total_negative = int(len(labels) - total_positive)
    if total_positive == 0 or total_negative == 0:
        raise ValueError("ROC curve requires both positive and negative samples.")

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_positive = positives[order].astype(np.int32)

    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_positive)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    tpr = true_positives.astype(np.float64) / float(total_positive)
    fpr = false_positives.astype(np.float64) / float(total_negative)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def binary_precision_recall_curve(labels, scores, positive_label=1):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    positives = labels == positive_label
    total_positive = int(np.sum(positives))
    if total_positive == 0:
        raise ValueError("Precision-recall curve requires positive samples.")

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_positive = positives[order].astype(np.int32)

    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_positive)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    precision = true_positives.astype(np.float64) / np.maximum(
        true_positives + false_positives, 1
    )
    recall = true_positives.astype(np.float64) / float(total_positive)
    thresholds = sorted_scores[threshold_indices]

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    average_precision = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return precision, recall, thresholds, average_precision
