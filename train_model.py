import pandas as pd
from pycaret.regression import setup, compare_models, create_model, tune_model, finalize_model, save_model

def train_and_save_model(csv_path="coffee_sales.csv", model_name="coffee_sales_rf_model"):
    df = pd.read_csv(csv_path)

    reg = setup(
        data = df,
        target = 'money',
        session_id = 123,
        fold_strategy = 'kfold',
        fold = 5,
        numeric_features = ['hour_of_day'],
        categorical_features = ['coffee_name','Time_of_Day','Weekday','Month_name'],
        normalize = True,
        transformation = True,
        transform_target = False,
        silent=True,
        verbose=False
    )

    best_model = compare_models(sort='MAE')

    rf = create_model('coffee_sales_rf_model')
    tuned_rf = tune_model(rf, optimize='MAE')

    final_rf = finalize_model(tuned_rf)

    save_model(final_rf, model_name)

    print(f"✔ Model retrained and saved as {model_name}.pkl")
    return model_name