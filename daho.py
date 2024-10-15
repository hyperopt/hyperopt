import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
def load_data():
    # Example with Boston Housing dataset (You can replace it with any dataset)
    from sklearn.datasets import load_boston
    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MEDV'] = data.target
    return df

# Step 2: Define a training function
def train_model(config):
    # Load data
    df = load_data()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['MEDV']), df['MEDV'], test_size=0.2)

    # Create the DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for XGBoost
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": config["eta"],
        "max_depth": int(config["max_depth"]),
        "min_child_weight": int(config["min_child_weight"]),
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"]
    }

    # Train the model
    result = xgb.train(params, dtrain, evals=[(dtest, "eval")], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)

    # Evaluate model performance
    predictions = result.predict(dtest)
    mse = mean_squared_error(y_test, predictions)
    tune.report(mean_squared_error=mse)

# Step 3: Hyperparameter search space
def get_search_space():
    return {
        "eta": tune.loguniform(1e-4, 1e-1),
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.uniform(1, 10),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0)
    }

# Step 4: Configure Ray Tune scheduler for distributed tuning
def run_hyperparameter_optimization():
    # Define ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="mean_squared_error",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )

    # Run Ray Tune with asynchronous hyperparameter search
    analysis = tune.run(
        train_model,
        config=get_search_space(),
        num_samples=50,  # Number of hyperparameter combinations to try
        scheduler=scheduler,
        resources_per_trial={"cpu": 1},  # Adjust based on available resources
        local_dir="./ray_results",  # Directory to store results
        verbose=1
    )

    # Print the best hyperparameters
    print("Best hyperparameters found were: ", analysis.best_config)

# Main function
if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Run the hyperparameter optimization
    run_hyperparameter_optimization()
    
    # Shutdown Ray when finished
    ray.shutdown()
