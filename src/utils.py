"""General utility functions."""

import sklearn
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def aupr(y_true, y_pred):
    """
    Compute the area under the precision-recall curve (AUPR).

    This function takes the true labels and predicted probabilities as input,
    computes the precision-recall curve, and calculates the area under curve.

    :param y_true: True binary labels
    :param y_pred: Estimated probabilities or decision function
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr = float(auc(recall, precision))
    print("precision recall: " + str(pr))


def evaluate_model(
    validation,
    cutoff_for_positive,
    actual_col="class",
    predicted_col="ACTIVITY"
):
    """
    Evaluate the performance of a binary classification model.

    This function takes a validation dataset, a cutoff value for positives,
    and the names of columns with actual labels and predicted probabilities.
    It computes the auPR, recall, and precision scores.

    :param validation: DataFrame containing validation data
    :param cutoff_for_positive: Cutoff value for defining positive predictions
    :param actual_col: Name of the column containing the actual class labels
        (default: "class")
    :param predicted_col: Name of the column with the predicted probabilities
        (default: "ACTIVITY")
    """
    actual = list(validation[actual_col])
    predicted = list(validation[predicted_col])
    aupr(actual, predicted)

    predicted_bin = [
        1.0 if x > cutoff_for_positive else 0.0 for x in list(predicted)]
    print("recall: ")
    print(sklearn.metrics.recall_score(actual, predicted_bin))
    print("precision: ")
    print(sklearn.metrics.precision_score(actual, predicted_bin))
