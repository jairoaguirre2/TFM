# Librerías estándar
import os
from time import time
import copy
import json
import pickle

# Manejo de datos
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingRegressor)
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

# Funciones propias de codificación
from funciones import ( encode_blosum_transition, encode_blosum, encode_b_fd, 
                       encode_b_d, encode_b_t, encode_b_c)

# Funciones propias de modelado
from modelos import (param_regre, prepare_regressor, prepare_data, train_model, 
                     evaluated_model, tablas_metricas, graficar_modelos, param_class, 
                     prepare_classifier)

SEED=164

# ===================================================================
# ENTRENAMIENTO COMPLETO DE MODELOS
# Incluye:
# 1. Dataset reducido
# 2. Codificación de secuencias (BLOSUM + FDT)
# 3. Entrenamiento de clasificadores y regresores
# 4. Reentrenamiento de modelos seleccionados (AdaBoost, GradientBoost)
# 5. Guardado de modelos y métricas para inferencia y ranking de mutantes
# ===================================================================

# Generamos función para extraer aa en posiciones mutadas

def extract_positions(seq):
    positions = [24, 31, 60, 157, 208, 213]
    seq_pos = "".join([seq[p] for p in positions])
    return seq_pos

# Creamos dataset reducido

dataset = pd.read_csv("dataset.csv", sep=";")
dataset["class"] = (dataset["fitness"] > 0).astype(int) 
dataset["wt_SM"]  = dataset["wt"].apply(lambda x: extract_positions(x))
dataset["mut_SM"] = dataset["mut"].apply(lambda x: extract_positions(x))

dataset.to_csv("dataset_reducido.csv", index=False)

# Codificamos modelos clasificadores

for func, max_length in zip([encode_b_d, encode_b_fd, encode_b_c, encode_b_t], [20, 6*20, 6*20*2, 6]):
    print(func)
    func(dataset, "wt_SM", "mut_SM", "class", "classifier_M", max_length=max_length)

# Generamos dataset para regresores y codificamos

data_reg = dataset[dataset["fitness"]>0.0] # filtramos
data_reg["log_fitness"] = np.log(data_reg["fitness"]) # transformamos target

for func, max_length in zip([encode_b_d, encode_b_fd, encode_b_c, encode_b_t], [20, 6*20, 6*20*2, 6]):
    print(func)
    func(data_reg, "wt_SM", "mut_SM", "log_fitness", "regressor_M", max_length=max_length)

# =======================================================================================================
# Tras las pruebas preelimnares de las codificaciones, se decide crear una nueva codificación
# Codificación FDT 
# =======================================================================================================


pd.concat([
    pd.read_csv("classifier_M_flaten_diff_blosum.csv").drop(columns=["class"]),
    pd.read_csv("classifier_M_trans_blosum.csv"),
], axis=1).to_csv("classifier_M_flatten_diff_trans_blosum.csv", index=False)


pd.concat([
    pd.read_csv("regressor_M_flaten_diff_blosum.csv").drop(columns=["log_fitness"]),
    pd.read_csv("regressor_M_trans_blosum.csv"),
], axis=1).to_csv("regressor_M_flatten_diff_trans_blosum.csv", index=False)

# ======================================================================================================
# Una vez generados los archivos csv codificados se procede con el entrenamiento 
# =======================================================================================================

# Primero se entrenan los modelos clasificadores 
 
dataset_files = [
    "classifier_M_concat_blosum.csv",
    "classifier_M_dif_blosum.csv",
    "classifier_M_flaten_diff_blosum.csv",
    "classifier_M_flatten_diff_trans_blosum.csv",
    "classifier_M_trans_blosum.csv"
]


params = param_class
train_function = prepare_classifier

np.random.seed(SEED)
seeds = [np.random.randint(0, 10000) for _ in range(3)]

for dataset in dataset_files:

    print(f"\n Entrenando dataset: {dataset} \n")
    
    # Cargamos dataset
    df = pd.read_csv(dataset)

    # Diccionario para este dataset
    train_results_all = {}
    start_total = time()
    
    # Bucle sobre semillas y modelos
    for seed in seeds:
        # Preparamos los datos
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data=df,
            features=[c for c in df.columns if c not in ['class', 'fitness']],
            target='class',
            scale_data=True,
            seed=seed
        )

        for model_cls, params_grid in params.items():
            print(model_cls)
            try:
                train_results = train_model(
                    model=model_cls(),
                    param_grid_to_use=params_grid,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    is_class=True,
                    seed=seed
                )

                model_trained, metrics_df, metrics_list = evaluated_model(
                    model=train_results["model"],
                    y_train=y_train,
                    y_test=y_test,
                    y_pred_cv=train_results["y_pred_cv"],
                    y_pred_test=train_results["y_pred_test"],
                    is_class=True
                )

                key = f"{model_cls.__name__}_seed{seed}"

                train_results_all[key] = {
                    "train_results": train_results,
                    "metrics": metrics_list,
                    "scaler": scaler, 
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test
                }

            except Exception as e:
                print(f"ERROR en {model_cls.__name__} con seed {seed}: {e}")

   
    #  Guardamos resultados por dataset
   
    base_name = dataset.replace(".csv", "")

    with open(f"{base_name}_train_results.pkl", "wb") as f:
        pickle.dump(train_results_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = (time() - start_total) / 60.0
    with open(f"{base_name}_train_time.txt", "w") as f:
        f.write(f"Tiempo total: {total_time:.2f} minutos\n")
        f.write(f"Modelos entrenados: {len(train_results_all)}\n")

    print(f"Completado: {dataset}")

# Generamos métricas y gráficas  

pickles = [
    "classifier_M_concat_blosum_train_results.pkl",
    "classifier_M_dif_blosum_train_results.pkl",
    "classifier_M_flaten_diff_blosum_train_results.pkl",
    "classifier_M_trans_blosum_train_results.pkl",
    "classifier_M_flatten_diff_trans_blosum_train_results.pkl",

]

for pkl_file in pickles:
    output_excel = pkl_file.replace("_train_results.pkl", ".xlsx")
    df_cv, df_test = tablas_metricas(pkl_file, output_excel=output_excel)
    print(f"{output_excel} generado")


save_dir = "graficas_class"
os.makedirs(save_dir, exist_ok=True)

for pkl_file in pickles:
    print(f"Generando gráficas para {pkl_file}...")
    graficar_modelos(pkl_file, is_class=True, save_dir=save_dir)

# ===================================================================================
# ===================================================================================

# Segundo, se entrenan los modelos clasificadores 

# Iniciamos entrenamiento 
dataset_files_f = [
    "regressor_M_concat_blosum.csv",
    "regressor_M_dif_blosum.csv",
    "regressor_M_flaten_diff_blosum.csv",
    "regressor_M_flatten_diff_trans_blosum.csv",
]

params = param_regre
train_function = prepare_regressor

np.random.seed(SEED)
seeds = [np.random.randint(0, 10000) for _ in range(3)]

for dataset in dataset_files_f:

    print(f"\n Entrenando dataset: {dataset} \n")
    
    # Cargamos dataset
    df = pd.read_csv(dataset)

    features = [c for c in df.columns if c not in ['log_fitness']]
    X = df[features].to_numpy()
    y = df["log_fitness"].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train = X_test = X
    y_train = y_test = y

    # Diccionario para este dataset
    train_results_all = {}

    start_total = time()

    # Bucle sobre semillas y modelos
    for seed in seeds:
        for model_cls, params_grid in params.items():
            print(seed, model_cls)
            try:
                train_results = train_model(
                    model=model_cls(),
                    param_grid_to_use=params_grid,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    cv_split=len(y_train),
                    is_class=False,
                    seed=seed
                )

                model_trained, metrics_df, metrics_list = evaluated_model(
                    model=train_results["model"],
                    y_train=y_train,
                    y_test=y_test,
                    y_pred_cv=train_results["y_pred_cv"],
                    y_pred_test=train_results["y_pred_test"],
                    is_class=False
                )

                key = f"{model_cls.__name__}_seed{seed}"

                train_results_all[key] = {
                    "train_results": train_results,
                    "metrics": metrics_list,
                    "scaler": scaler,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test
                }

            except Exception as e:
                print(f"ERROR en {model_cls.__name__} con seed {seed}: {e}")

   
    #  Guardamos resultados por dataset
   
    base_name = dataset.replace(".csv", "")

    with open(f"{base_name}_train_results.pkl", "wb") as f:
        pickle.dump(train_results_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = (time() - start_total) / 60.0
    with open(f"{base_name}_train_time.txt", "w") as f:
        f.write(f"Tiempo total: {total_time:.2f} minutos\n")
        f.write(f"Modelos entrenados: {len(train_results_all)}\n")

    print(f"Completado: {dataset}")

 
# Métricas y gráficas  

pickles = [
    "regressor_M_flaten_diff_blosum_train_results.pkl",
    "regressor_M_flatten_diff_trans_blosum_train_results.pkl",
    "regressor_M_dif_blosum_train_results.pkl",
    "regressor_M_concat_blosum_train_results.pkl"
    ]

for pkl_file in pickles:
    output_excel = pkl_file.replace("_train_results.pkl", ".xlsx")
    df_cv, df_test = tablas_metricas(pkl_file, output_excel=output_excel)
    print(f"{output_excel} generado")

# Carpeta donde guardar las gráficas
save_dir = "graficas_reg"
os.makedirs(save_dir, exist_ok=True)

for pkl_file in pickles:
    print(f"Generando gráficas para {pkl_file}...")
    graficar_modelos(pkl_file, is_class=False, save_dir=save_dir)


# =========================================================================================================
# Se visualizaron gráficas y tablas de métricas y los modelos seleccionados fueron 
# Adaboost para clasificadores y GradientBoost para regresores
# Una vez tenemos los modelos seleccionados los reentrenamos conservando los mejores
# hiperparámetros previamente optimizados.
# ==========================================================================================================

"""
Dataset: Blossum Flatten Diferencia Transition
CV: 3 seeds x 5 folds
Estimator: AdaBoost 
"""
# Cargamos resultados previos
with open("classifier_M_flatten_diff_trans_blosum_train_results.pkl", "rb") as f:    
    data = pickle.load(f)

# Seleccionamos modelos Adaboost
trained_models = [m  for m in list(data.keys()) if m.startswith("Ada")]

# Reentrenamos con CV
metrics_total = {}
models = []


bas_total = []

for model in trained_models:
    print(model)
    seed = int(model.split("_")[-1][4:])

    params = data[model]["train_results"]["best_params"]
    scaler = data[model]["scaler"]


    X_train, X_test =  data[model]["X_train"], data[model]["X_test"]
    y_train, y_test =  data[model]["y_train"], data[model]["y_test"]

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

  
    estimator = AdaBoostClassifier(random_state=seed, **params)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    metrics = []

    for train, test in cv.split(X, y):
        estimator.fit(X[train,:], y[train])
        pred = estimator.predict(X[test,:])
        bas = balanced_accuracy_score(y[test], pred)
        metrics.append(bas)
        models.append({"estimator": copy.deepcopy(estimator), "scaler":scaler, "bas":bas})
        print(bas)
        bas_total.append(bas)

    metrics_total[model] = {
        "bas": metrics,
        "bas_avg": sum(metrics) / len(metrics),
    }

# Guardamos resultados
metrics_total["bas_total"] = sum(bas_total) / len(bas_total)

with open("models_adam_for_inference_metrics_hotspots.json", "wt") as f:
    json.dump(metrics_total, f, indent=4)

with open("models_adam_for_inference_hotspots.pkl", "wb") as f:
    pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

print(models)
print(len(models))
print(metrics_total)

# =================================================================================================
# Una vez tenemos reentrenados Adaboost, procedemos con GradientBoost
# ==================================================================================================

"""
Dataset: Dataset: Blossum Flatten Diferencia Transition
CV: 3 seeds x LOO
Estimator: GradientBoosting 
"""

# Cargamos resultados previos
with open("regressor_M_flatten_diff_trans_blosum_train_results.pkl", "rb") as f:
    data = pickle.load(f)

# Seleccionamos modelos GradientBoost
trained_models = [m  for m in list(data.keys()) if m.startswith("Gra")]

metrics_total = {}
models = []

# Reentrenamos con LOO
sp_total = []

for model in trained_models:
    print(model)
    seed = int(model.split("_")[-1][4:])

    params = data[model]["train_results"]["best_params"]
    scaler = data[model]["scaler"]

    X_train, X_test =  data[model]["X_train"], data[model]["X_test"]
    y_train, y_test =  data[model]["y_train"], data[model]["y_test"]

    X = X_train
    y = y_train

    estimator = GradientBoostingRegressor(random_state=seed, **params)

    cv = KFold(
        n_splits=len(y),
        shuffle=True,
        random_state=seed,
    )
    metrics = []
    predictions = np.zeros(len(y))
    for train, test in cv.split(X, y):
        estimator.fit(X[train,:], y[train])
        pred = estimator.predict(X[test,:])
        predictions[test] = pred
        models.append({"estimator": copy.deepcopy(estimator), "scaler":scaler})

    sp = spearmanr(y, predictions).statistic
    print(sp)
    sp_total.append(sp)
    metrics_total[model] = {
        "sp": sp,
    }

metrics_total["sp_total"] = sum(sp_total) / len(sp_total)

with open("models_grad_regressor_for_inference_metrics_hotspots.json", "wt") as f:
    json.dump(metrics_total, f, indent=4)

with open("models_grad_regressor_for_inference_hotspots.pkl", "wb") as f:
    pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

print(len(models))
print(metrics_total)