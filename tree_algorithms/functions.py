from collections import Counter
import numpy as np
import pandas as pd


def make_column_numeric(df, column_index):
    """
    Transforms non-numeric column to numeric.

    :param df: Input dataset
    :param column_index: Index of column which will be transformed.
    :return: Returns nothing.
    """
    col = df.iloc[:, column_index]
    c = Counter(col)
    mapped = dict(zip(c, range(len(c))))
    new_col = np.array([mapped[val] for val in col])
    df.iloc[:, column_index] = new_col


def make_test_data(train_df, target_df, n):
    """
    Randomly generates n-sized test data from train and target dataset.

    :param train_df: Train dataset.
    :param target_df: Target dataset.
    :param n: Size of desired test dataset.
    :return: Returns transformed datasets.
    """
    possible_indexes = np.arange(train_df.shape[0])
    test_index = np.random.choice(possible_indexes, n, replace=False)
    test_df = train_df.loc[test_index]
    test_df.loc[:, ["target"]] = target_df.loc[test_index]
    new_train_df = train_df.drop(labels=test_index, axis=0)
    new_target_df = target_df.drop(labels=test_index, axis=0,)
    new_train_df = new_train_df.reset_index(drop=True)
    new_target_df = new_target_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return new_train_df, new_target_df, test_df


def cross_validate(model, k=10, mode=None, model_type="regression", min_sample_split=2,
                   train_df=None, target_df=None, full_dataset=None):
    """
    Cross validates data to pick the best model.

    :param model: Model used.
    :param k: validate data size.
    :param mode: Specifies cross_validation output type.
    :param model_type: Specifies whether cross-validation is used on regression.
    :param min_sample_split: Parameter for creating model object.
    :param train_df: Training dataset.
    :param target_df: Target dataset.
    :param full_dataset: Dataset with merged training and target dataset.
    :return: Returns best model object.
    """
    if full_dataset is None:
        dataset = pd.concat([train_df, target_df], axis=1)
    else:
        dataset = full_dataset
    sample_amount = dataset.shape[0]
    models = []
    for i in range(0, sample_amount, k):
        m = model(min_sample_split=min_sample_split)
        validate_data_indexes = np.arange(i, i+k if i+k <= sample_amount else sample_amount)
        validate_data = dataset.iloc[validate_data_indexes]
        new_dataset = dataset.drop(validate_data_indexes, axis=0)
        train_df, target_df = new_dataset.iloc[:, :-1], new_dataset.iloc[:, -1]
        if mode is not None:
            m.fit(train_df, target_df, mode=mode)
        else:
            m.fit(train_df, target_df)
        predictions = m.predict(validate_data)
        if mode == "best":
            models.append((m.prediction_score(predictions, validate_data), m))
        else:
            models.append((validate_data, m))

    if mode == "best":
        if model_type == "regression":
            min_rss_model = min(models, key=lambda x: x[0])
            best_model = min_rss_model
        else:
            max_prediction_percentage = max(models, key=lambda x: x[0])
            best_model = max_prediction_percentage
        return best_model[1]
    else:
        return models


def generate_nan(train_dataset, n):
    """Function used to create random nan for test purposes."""

    data = train_dataset.to_numpy()
    for _ in range(n):
        sample_index, feature_index = np.random.randint(data.shape[0]), np.random.randint(data.shape[1])
        data[sample_index, feature_index] = np.nan
    return pd.DataFrame(data, columns=train_dataset.columns)
