import plotly.graph_objects as go
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import pycountry
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import torch
import mlflow.pytorch
from pysr import PySRRegressor
import optuna
import random
import numpy as np

def job_type_label(title):
    if 'Robotics' in title:
        return 'Robotics'
    elif 'Machine Learning' in title:
        return 'ML'
    elif "ML" in title:
        return 'ML'
    elif 'AI' in title:
        return 'AI'
    elif 'Deep Learning' in title:
        return 'AI'
    elif 'NLP' in title:
        return 'AI'
    elif 'Data' in title:
        return 'Data'
    elif 'Insight' in title:
        return 'Data'
    elif 'Analytics' in title:
        return 'Data'
    elif 'Business Intelligence' in title:
        return 'BI'
    elif 'BI' in title:
        return 'BI'
    else:
        return "Others"

def leadership_label(title):
    if 'Manager' in title:
        return 'Manager'
    elif 'Lead' in title:
        return 'Manager'
    elif 'Head' in title:
        return 'Head'
    else:
        return 'Staff'

def get_counts(df, column, order=None, top_n=20):
    counts = df[column].value_counts()
    if order:
        counts = counts.reindex(order, fill_value=0)
    counts = counts.head(top_n)
    return counts.index.tolist(), counts.values.tolist()

def plot_box_by_category(fig, df, category_col, value_col, color_dict, row, col):
    for cat, color in color_dict.items():
        vals = df.loc[df[category_col] == cat, value_col]
        if not vals.empty:
            fig.add_trace(
                go.Box(y=vals, name=cat, marker_color=color),
                row=row, col=col
            )

def heatmap_by_category(df, cat1, cat2, value, color, colorlabel,
                        xbar_position, ybar_position, x_order=None, y_order=None):
    subset_df = df.groupby([cat1, cat2])[value].median().reset_index()
    pivot_tab = subset_df.pivot(index=cat1, columns=cat2, values=value)
    pivot_tab = pivot_tab.reindex(index=y_order, columns=x_order)
    heatmap = go.Heatmap(
        z=pivot_tab.values,
        x=pivot_tab.columns,
        y=pivot_tab.index,
        colorscale=color,
        colorbar=dict(x=xbar_position, y=ybar_position, len=0.4, title=colorlabel),
        text=pivot_tab.values,
        texttemplate="%{text}"
        )
    return heatmap

def features_preprocessing(df, ordinal_cols, nominal_cols, stratify_col):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    ct = ColumnTransformer(transformers=[('ord', OrdinalEncoder(), ordinal_cols),
                                        ('nom', OneHotEncoder(sparse_output=False), nominal_cols)],
                                        remainder="passthrough")
    X = ct.fit_transform(X)
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=df[[stratify_col]])
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return ct, sc, X_train, X_test, y_train, y_test

def reg_metrics(df, reg_model, column_transformer, X_train, y_train, X_test, y_test,
                feature_importance=True, usd_scale_back=True, get_model=None):
    reg_model.fit(X_train, y_train)
    r2 = reg_model.score(X_test, y_test)
    y_pred = reg_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R^2 Score: {r2:.3f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"MAE: {mae:,.2f}")

    mse_usd = None
    rmse_usd = None
    mae_usd = None
    if usd_scale_back:
        y_true_usd = np.exp(y_test)
        y_pred_usd = np.exp(y_pred)
        mse_usd = mean_squared_error(y_true_usd, y_pred_usd)
        rmse_usd = root_mean_squared_error(y_true_usd, y_pred_usd)
        mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)
        print(f"RMSE in usd: {rmse_usd:,.2f}")
        print(f"MSE in usd: {mse_usd:,.2f}")
        print(f"MAE in usd: {mae_usd:,.2f}")

    importances_series = None
    if feature_importance:
        try:
            feature_names_in = df.columns[:-1]
            feature_names_out = column_transformer.get_feature_names_out(feature_names_in)
            importances = reg_model.feature_importances_
            importances_series = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)
            print(importances_series)
        except AttributeError:
            print("This model does not have feature_importances_ attribute.")
        except Exception as e:
            print(f"Error getting feature importances: {e}")
    if get_model:
        return r2, rmse, mse, mae, mse_usd, rmse_usd, mae_usd, importances_series, reg_model
    else:
        return r2, rmse, mse, mae, mse_usd, rmse_usd, mae_usd, importances_series


def alpha2_to_alpha3(alpha2):
        return pycountry.countries.get(alpha_2=alpha2).alpha_3

def log_to_mlflow(model, run_name, model_name, r2, rmse, mse, mae):
    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_params(model.get_params())
        # Log metrics
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        # Log the model itself
        mlflow.sklearn.log_model(model, model_name)

def log_ann_to_mlflow(model, run_name, model_name, r2, rmse, mse, mae, params=None):
    """
    Logs a PyTorch ANN model and metrics to MLflow.

    Args:
        model: The trained PyTorch model.
        run_name (str): Name of the MLflow run.
        model_name (str): Name to register the model under in MLflow.
        r2, rmse, mse, mae: Evaluation metrics (floats).
        params (dict or None): Optional dict of parameters to log (e.g., epochs, lr, hidden sizes).
    """
    with mlflow.start_run(run_name=run_name):
        if params is not None:
            mlflow.log_params(params)
        else:
            mlflow.log_param("model_architecture", model.__class__.__name__)
            mlflow.log_param("total_params", sum(p.numel() for p in model.parameters()))

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        mlflow.pytorch.log_model(model, model_name)

def train_nn_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    train_losses = []
    test_losses = []

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Test Loss: {epoch_test_loss:.4f}")

    return train_losses, test_losses

def sr_tuning(X_train, y_train, X_test, y_test, output_path, n_trials=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def objective(trial):
        # Suggest values within ranges:
        niterations = trial.suggest_int('niterations', 50, 300)
        population_size = trial.suggest_int('population_size', 50, 300)
        populations = trial.suggest_int('populations', 10, 100)
        maxsize = trial.suggest_int('maxsize', 25, 100)
        parsimony = trial.suggest_float('parsimony', 0.0, 0.2)
        ncycles_per_iteration = trial.suggest_int('ncycles_per_iteration', 500, 2000)

        # Instantiate PySRRegressor with tunable parameters
        model = PySRRegressor(
            random_state=seed,
            model_selection='accuracy',
            niterations=niterations,
            population_size=population_size,
            populations=populations,
            maxsize=maxsize,
            parsimony=parsimony,
            ncycles_per_iteration=ncycles_per_iteration,
            binary_operators=['+', '*', '-', '/'],
            unary_operators=['cos', 'exp', 'sin', 'log'],
            elementwise_loss="loss(x, y) = (x - y)^2",
            batching=True,
            bumper=True,
            verbosity=0,
            progress=False,
            procs=16,
            output_directory=output_path
        )

        # Fit on training data
        model.fit(X_train, y_train)

        # Evaluate predictive loss
        y_pred = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred)
        return test_loss

    # Create and run Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Get best hyperparameters
    best_params = study.best_params
    best_loss = study.best_value

    # Retrain best model on all data (optional)
    best_model = PySRRegressor(
        random_state=seed,
        model_selection='accuracy',
        binary_operators=['+', '*', '-', '/'],
        unary_operators=['cos', 'exp', 'sin', 'log'],
        elementwise_loss="loss(x, y) = (x - y)^2",
        batching=True,
        bumper=True,
        verbosity=0,
        progress=False,
        procs=16,
        output_directory=output_path,
        **best_params
    )
    best_model.fit(X_train, y_train)

    return study, best_model, best_loss

