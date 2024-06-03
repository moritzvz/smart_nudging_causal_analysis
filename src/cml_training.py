import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from econml.dml import CausalForestDML


def train_model(train_data_path='data/train_data.csv'):
    """
    Tunes and trains a Causal Forest in a Double Machine Learning Framework
    on the simulated training data.

    Parameters:
    train_data_path (str): Path to the CSV file containing the training data.

    Returns:
    None: The trained model is saved to a file.
    """
    train_df = pd.read_csv(train_data_path)
    y_train, t_train, x_train = train_df["Y"], train_df["T"], train_df.drop(columns=["T", "Y"])
    rs = 2024

    # final stage model (first stage automatic)
    est = CausalForestDML(discrete_treatment=True, discrete_outcome=True)  # model_t=model_t, model_y=model_y,
    tune_param = {
        'max_samples': [.4, .45],
        'min_balancedness_tol': [.3, .4, .5],
        'min_samples_leaf': [15, 30, 45],
        'max_depth': [None, 5, 7],
        'min_var_fraction_leaf': [None, .01],
    }
    print(f"Tuning: {tune_param}\n")
    est.tune(Y=y_train, T=t_train, X=x_train, params=tune_param, )
    tuning_result_param = f"max_samples: {est.max_samples}, min_balancedness_tol: {est.min_balancedness_tol}," \
                          f" min_samples_leaf: {est.min_samples_leaf}, max_depth: {est.max_depth}, " \
                          f"min_var_fraction_leaf: {est.min_var_fraction_leaf}, n_estimators: {est.n_estimators}"
    print(f"Tuned to: {tuning_result_param}")

    # fit causal forest
    est.fit(Y=y_train, T=t_train, X=x_train)

    joblib.dump(est, 'model/causal_forest_dml_model.pkl')
    print("Model training complete. Model saved to model/causal_forest_dml_model.pkl")


if __name__ == "__main__":
    train_model()