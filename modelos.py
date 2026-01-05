import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import pickle
import os
import math

from sklearn.model_selection import (
    train_test_split, cross_val_predict, GridSearchCV, StratifiedKFold, KFold)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix)

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor,
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from sklearn.base import is_classifier, is_regressor
from scipy.stats import spearmanr

# =====================================================================================================================
# FUNCIONES PARA PREPARAR DATOS, ENTRENAR Y EVALUAR MODELOS
# =====================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------
# Preparar datos
# -----------------------------------------------------------------------------------------------------------------------------

def prepare_data(data, features, target, test_size=0.2, scale_data=True, seed=164):
    """
    Divide un DataFrame en conjuntos de entrenamiento y test, con opción de escalar los datos.

    Args:
        data (pd.DataFrame): DataFrame con los datos completos.
        features (list): Lista de columnas de características.
        target (str): Nombre de la columna objetivo.
        test_size (float, optional): Fracción del dataset para test. Defaults to 0.2.
        scale_data (bool, optional): Si True, aplica StandardScaler. Defaults to True.
        seed (int, optional): Semilla para reproducibilidad. Defaults to 164.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# -----------------------------------------------------------------------------------------------------------------------------
# Param grids para regresión y clasificación
# -----------------------------------------------------------------------------------------------------------------------------

param_regre = {
    KNeighborsRegressor: {
        "n_neighbors": [3, 5, 7], 
        "weights": ["uniform", "distance"], 
        "p": [1, 2]
    },
    RandomForestRegressor: {
        "n_estimators": [50, 100, 200], 
        "max_depth": [None, 5, 10],  
        "min_samples_split": [2, 5], 
        "min_samples_leaf": [1, 2] 
    },
    GradientBoostingRegressor: {
        "n_estimators": [100, 200],  
        "learning_rate": [0.01, 0.1], 
        "max_depth": [3, 5], 
        "subsample": [0.8, 1.0] 
    },
    AdaBoostRegressor: {
        "n_estimators": [50, 100], 
        "learning_rate": [0.01, 0.1 , 1.0] 
    },
    ExtraTreesRegressor: {
        "n_estimators": [50, 100], 
        "max_depth": [None, 5, 10],
        "min_samples_split": [2,5] 
    },
    MLPRegressor: {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)], 
        "activation": ["relu", "tanh"], 
        "learning_rate_init": [0.001, 0.01],
        "max_iter":[600],
    },
    SVR: {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"], 
        "gamma": ["scale", "auto"] 
    }
}

param_class = {
    KNeighborsClassifier: {
        "n_neighbors": [3, 5, 7], 
        "weights": ["uniform", "distance"], 
        "p": [1, 2] 
    },
    RandomForestClassifier: {
        "n_estimators": [50, 100, 200], 
        "max_depth": [None, 5, 10], 
        "min_samples_split": [2, 5], 
        "min_samples_leaf": [1, 2], 
        "criterion": ["gini", "entropy", "log_loss"]
    },
    GradientBoostingClassifier: {
        "n_estimators": [100, 200], 
        "learning_rate": [0.01, 0.1], 
        "max_depth": [3, 5], 
        "subsample": [0.8, 1]
    },
    AdaBoostClassifier: {
        "n_estimators": [50, 100], 
        "learning_rate": [0.01, 0.1, 1.0] 
    },
    ExtraTreesClassifier: {
        "n_estimators": [150, 300], 
        "max_depth": [None, 5, 10], 
        "min_samples_split": [2, 5], 
        "criterion": ["gini", "entropy", "log_loss"] 
    },
    MLPClassifier: {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)], 
        "activation": ["relu", "tanh"], 
        "learning_rate_init": [0.001, 0.01],
        "max_iter":[600],
    },
    SVC: {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"], 
        "gamma": ["scale", "auto"] 
    }
}

# -----------------------------------------------------------------------------------------------------------------------------
# Preparar modelo regresor y clasificador
# -----------------------------------------------------------------------------------------------------------------------------

def prepare_regressor(model_reg, param_grid = param_regre ):
    """
    Prepara un modelo regresor y retorna su grid de hiperparámetros.

    Args:
        model_reg (sklearn regressor): Clase del modelo regresor.
        param_grid (dict, optional): Diccionario con grids de hiperparámetros. Defaults to param_regre.

    Returns:
        tuple: modelo instanciado, grid de hiperparámetros
    """
    model = model_reg()
    best_param_regre = param_grid.get(model_reg, None)
    return model, best_param_regre

def prepare_classifier(model_class, param_grid=param_class):
    """
    Prepara un modelo clasificador y retorna su grid de hiperparámetros.

    Args:
        model_class (sklearn classifier): Clase del modelo clasificador.
        param_grid (dict, optional): Diccionario con grids de hiperparámetros. Defaults to param_class.

    Returns:
        tuple: modelo instanciado, grid de hiperparámetros
    """
    model = model_class()
    best_param_class = param_grid.get(model_class, None)
    return model, best_param_class

# -----------------------------------------------------------------------------------------------------------------------------
# Entrenar modelo
# -----------------------------------------------------------------------------------------------------------------------------

def train_model(model, param_grid_to_use, X_train, y_train, X_test,
                cv_split=5, is_class=False, seed=164,
                save_path=None):
    """
    Entrena un modelo con GridSearchCV opcional y devuelve predicciones, mejor modelo y parámetros.

    Args:
        model (sklearn estimator): Modelo instanciado.
        param_grid_to_use (dict): Grid de hiperparámetros para GridSearchCV.
        X_train, y_train, X_test: Datos de entrenamiento y test.
        cv_split (int, optional): Número de folds CV. Defaults to 5.
        is_class (bool, optional): Si True, el modelo es de clasificación. Defaults to False.
        seed (int, optional): Semilla para reproducibilidad. Defaults to 164.
        save_path (str, optional): Ruta para guardar resultados en pickle. Defaults to None.

    Returns:
        dict: Diccionario con modelo, mejores parámetros y predicciones.
    """
    best_params = None
    cv_base = StratifiedKFold if is_class else KFold
    cv = cv_base(n_splits=cv_split, 
                 shuffle=True, random_state=seed)

    if param_grid_to_use is not None:
        scoring = "accuracy" if is_class else "neg_mean_squared_error"
        grid = GridSearchCV(
            model, param_grid=param_grid_to_use, cv=cv, 
            scoring=scoring, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_ 
    else:
        best_model = model

    y_pred_cv = cross_val_predict(
        best_model, X_train, y_train, cv=cv)
     
    best_model.fit(X_train, y_train) 
    y_pred_test = best_model.predict(X_test)

    train_results = {
        "model": best_model,
        "best_params": best_params,
        "y_pred_cv": y_pred_cv,
        "y_pred_test": y_pred_test,
        "y_train": y_train,
        "y_test": X_test  
    } 

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(train_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_results    

# -----------------------------------------------------------------------------------------------------------------------------
# Evaluar modelo
# -----------------------------------------------------------------------------------------------------------------------------

def evaluated_model(model, y_train, y_test=None, y_pred_cv=None, y_pred_test=None,
                    is_class=False, pickle_path=None):
    """
    Evalúa un modelo con métricas de regresión o clasificación y actualiza pickle si se indica.

    Args:
        model: modelo entrenado
        y_train, y_test: valores reales
        y_pred_cv, y_pred_test: predicciones
        is_class (bool): True si es clasificación
        pickle_path (str): ruta a pickle para actualizar métricas

    Returns:
        tuple: modelo, DataFrame de métricas, diccionario de métricas
    """
    metrics_list = {}

    # Evaluar CV
    if y_pred_cv is not None:
        if is_class:
            metrics_cv = {
                "model": model.__class__.__name__,
                "dataset": "CV_train",
                "accuracy": accuracy_score(y_train, y_pred_cv),
                "precision": precision_score(y_train, y_pred_cv, average='weighted'),
                "recall": recall_score(y_train, y_pred_cv, average='weighted'),
                "f1": f1_score(y_train, y_pred_cv, average='weighted'),
                "balanced_accuracy": balanced_accuracy_score(y_train, y_pred_cv),
                "confusion_matrix": confusion_matrix(y_train, y_pred_cv).tolist()
            }
        else:
            metrics_cv = {
                "model": model.__class__.__name__,
                "dataset": "CV_train",
                "MSE": mean_squared_error(y_train, y_pred_cv),
                "MAE": mean_absolute_error(y_train, y_pred_cv),
                "R2": r2_score(y_train, y_pred_cv),
                "Spearman": spearmanr(y_train, y_pred_cv).statistic
            }
        metrics_list["CV"] = metrics_cv

    # Evaluar Test
    if y_pred_test is not None and y_test is not None:
        if is_class:
            metrics_test = {
                "model": model.__class__.__name__,
                "dataset": "Test",
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test, average='weighted'),
                "recall": recall_score(y_test, y_pred_test, average='weighted'),
                "f1": f1_score(y_test, y_pred_test, average='weighted'),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
                "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist()
            }
        else:
            metrics_test = {
                "model": model.__class__.__name__,
                "dataset": "Test",
                "MSE": mean_squared_error(y_test, y_pred_test),
                "MAE": mean_absolute_error(y_test, y_pred_test),
                "R2": r2_score(y_test, y_pred_test),
                "Spearman": spearmanr(y_test, y_pred_test).statistic
            }
        metrics_list["Test"] = metrics_test

    metrics_df = pd.DataFrame(list(metrics_list.values()))

    # Guardar en pickle si se indica
    if pickle_path:
        with open(pickle_path, "rb") as f:
            train_results = pickle.load(f)
        train_results["evaluation"] = metrics_list
        train_results["evaluation_df"] = metrics_df
        with open(pickle_path, "wb") as f:
            pickle.dump(train_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model, metrics_df, metrics_list

# -----------------------------------------------------------------------------------------------------------------------------
# Tablas de métricas y promedio
# -----------------------------------------------------------------------------------------------------------------------------

def tablas_metricas(pickle_path, output_excel=None):
    """
    Genera tablas de métricas separadas y promedios a partir de un pickle de resultados.

    Args:
        pickle_path (str): Ruta del pickle que contiene resultados de entrenamiento.
        output_excel (str, optional): Ruta para guardar archivo Excel con métricas. Defaults to None.

    Returns:
        tuple: DataFrames de métricas CV y Test
    """
    with open(pickle_path, "rb") as f:
        train_results_all = pickle.load(f)

    cv_rows = {}
    test_rows = {}

    for i,(model_name, data) in enumerate(train_results_all.items()):
        metrics = data.get("evaluation", data.get("metrics", {}))

        if "CV" in metrics:
            cv_rows[i] = metrics["CV"].copy()
            cv_rows[i]["model_id"] = model_name

        if "Test" in metrics:
            test_rows[i] = metrics["Test"].copy()
            test_rows[i]["model_id"] = model_name

    df_cv = pd.DataFrame.from_dict(cv_rows, orient="index").sort_values(by="model")
    df_test = pd.DataFrame.from_dict(test_rows, orient="index").sort_values(by="model")

    df_cv_mean = df_cv.groupby("model", as_index=False).mean(numeric_only=True)
    df_test_mean = df_test.groupby("model", as_index=False).mean(numeric_only=True)

    if output_excel:
        if not os.path.exists(output_excel):
            with pd.ExcelWriter(output_excel, mode="w") as writer:
                df_cv.to_excel(writer, sheet_name="CV", index=False)

        with pd.ExcelWriter(output_excel, mode="a", if_sheet_exists="replace") as writer:
            df_test.to_excel(writer, sheet_name="Test", index=False)
            df_cv_mean.to_excel(writer, sheet_name="CV_mean", index=False)
            df_test_mean.to_excel(writer, sheet_name="Test_mean", index=False)

        print(f"Tablas generadas y guardadas")

    return df_cv, df_test

# -----------------------------------------------------------------------------------------------------------------------------
# Graficar modelos
# -----------------------------------------------------------------------------------------------------------------------------

def graficar_modelos(pickle_path, is_class=False, save_dir=None):
    """
    Genera gráficas de predicciones o matrices de confusión promedio por modelo.

    Args:
        pickle_path (str): Ruta del pickle con resultados.
        is_class (bool): True si es clasificación. Defaults to False.
        save_dir (str, optional): Directorio para guardar figuras. Defaults to None.
    """
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    modelos = {}
    for key, data in results.items():
        base = key.split("_seed")[0]
        if base not in modelos:
            modelos[base] = []
        modelos[base].append(data)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for dataset in ["CV", "Test"]:
        bases = list(modelos.keys())
        n = len(bases)
        ncols = 4
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()

        for idx, base in enumerate(bases):
            ax = axes[idx]
            y_true = modelos[base][0]["y_train"] if dataset=="CV" else modelos[base][0]["y_test"]

            preds = []
            cms = []

            for entrada in modelos[base]:
                tr = entrada["train_results"]
                y_pred = np.array(tr["y_pred_cv"] if dataset=="CV" else tr["y_pred_test"])
                preds.append(y_pred)

                if is_class:
                    y_t = entrada["y_train"] if dataset=="CV" else entrada["y_test"]
                    cms.append(confusion_matrix(y_t, y_pred))

            preds = np.column_stack(preds)
            y_pred_mean = preds.mean(axis=1)
            y_pred_std  = preds.std(axis=1)

            if not is_class:
                ax.errorbar(y_true, y_pred_mean, yerr=y_pred_std, fmt='o', alpha=0.7, markersize=3)
                minv = min(y_true.min(), y_pred_mean.min())
                maxv = max(y_true.max(), y_pred_mean.max())
                ax.plot([minv, maxv], [minv, maxv], 'r--')
                ax.set_title(base)
                ax.set_xlabel("y_true")
                ax.set_ylabel("y_pred_mean")
            else:
                cm_stack = np.stack(cms, axis=-1)
                cm_mean = cm_stack.mean(axis=-1)
                sns.heatmap(cm_mean, annot=True, fmt=".1f", cmap="Blues", ax=ax)
                ax.set_title(base)

        for j in range(idx+1, nrows*ncols):
            axes[j].axis("off")

        plt.suptitle(f"{dataset} — Modelos promedio", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir:
            filename = os.path.join(save_dir, f"{os.path.basename(pickle_path).replace('.pkl','')}_{dataset}.png")
            plt.savefig(filename, dpi=300)
            print(f"Figura guardada: {filename}")
