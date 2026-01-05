import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os 
from time import time
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
from modelos import (prepare_data, prepare_classifier, 
                     prepare_regressor, train_model, 
                     evaluated_model, param_regre, param_class,
                     tablas_metricas, graficar_modelos)
from funciones import (encode_blosum, encode_blosum_transition,
                       encode_b_c, encode_b_d, encode_b_t, blosum62,
                       encode_oh, encode_oh_c, encode_oh_d, encode_oh_t, mutations)

SEED=164

# ======================================================================================================================
# ARCHIVO DE PRUEBA Y DESARROLLO
# En este script se entrenaron modelos preliminares y se realizaron pruebas experimentales 
# Las funciones aquí utilizadas fueron posteriormente modificadas, optimizadas  y trasladadas a los módulos 
# finales empleados para la obtención de los resultados presentados en el apartado de Resultados.
# Este archivo se incluye con fines de trazabilidad del flujo de trabajo y documentación del proceso de desarrollo,
#  no como código final reproducible
# ====================================================================================================================

# Primer paso: Generar las secuencias de las posiciones mutadas 

seq = "ANPYERGPNPTDALLEASSGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSLAWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNHMINRASSTVRSRIDSSRLAVMGHSMGGGGTLRLASQRPDLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYVELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ"

muts = [
    "A25G_L32M_Y61F_T158V_D204C_N214M",
    "S33L_Y93G_S99R_T158P_E174Q_D204C_F209I_N214P_S249C",
    "R31S_T158A_D204C_F209Y_N214M",
    "A25C_L32M_T158P_D204C_F209W_N214T",
    "Y61F_D204C_F209E_N214F",
    "L32V_T158C_D204C_F209W_N214P",
    "L32M_Y61G_T158C_D204C_F209R_N214L",
    "A25K_L32A_Y61F_T158P_D204C_F209T_N214V",
    "S23C_A25H_T158I_D204C_F209A_N214V",
    "L32S_D204N",
    "A25K_L32S_Y61W_T158E_D204C_F209I_N214P",
    "L32Q_Y61F_D204C_F209V_N214M",
    "A25R_T158K_D204C_F209D_N214P",
    "S49N",
    "A25R_V29I_Y93G_D204C_F209I_N214P_S249C",
    "L32S_D204N",
    "A25E_L32G_Y61V_T158C_D204C_F209V_N214T",
    "F22S_R31S_L32W_Y61R_T158A_D204C_F209G_N214T",
    "A25T_R31S_L32R_Y61L_D204C_F209Q_N214E",
    "A25M_L32R_Y61M_T158C_D204C_F209V_N214T",
    "A25V_R31S_L32M_T158V_D204C_F209L_N214A",
    "A25S_L32G_Y61G_T158S_D204C_F209Y_N214V",
    "A25N_R31S_L32R_Y61Q_T158M_F209G_N214E",
    "A25D_R31S_Y61T_T158R_D204C_F209V_N214V",
    "A25P_L32P_Y61A_T158W_D204C_F209V_N214E",
    "A25R_L32V_Y61S_D204C_F209A_N214Y",
    "A25H_L32Q_Y61L_D204C_F209D_N214E",
    "A25K_L32G_Y61T_T158H_D204C_F209M_N214A",
    "A25K_L32A_Y61F_T158P_D204C_F209T_N214V",
    "A25W_L32F_Y61E_T158C_D204C_F209E_N214Q",
    "A25Q_L32G_Y61A_T158H_D204C_F209V_N214M",
    "A25R_L32S_Y61H_T158E_D204C_F209T_N214V",
    "A25C_L32G_Y61R_T158M_D204C_F209S_N214K",
    "S23C_A25H_T158I_D204C_F209A_N214V",
    "A25Q_L32V_Y61M_T158C_D204C_F209A_N214F",
    "A25P_L32R_Y61W_T158I_D204C_F209V_N214G",
    "L32S_D204N",
    "L32E_Y61L_T158Y_D204C_F209K_N214P",
    "A25S_L32G_T158Y_D204C_F209V_N214T",
    "L32Q_Y61P_T158Y_D204C_F209V_N214V",
    "A25F_R31S_L32M_Y61C_T158D_D204C_F209Q_N214D",
    "L32R_Y61T_T158E_D204C_F209V_N214Q",
    "A25R_L32R_Y61P_T158Y_D204C_F209E_N214V",
    "L32Q_Y61P_T158Y_D204C_F209V_N214V",
    "A25M_R31S_L32V_Y61A_T158I_D204C_F209L_N214V",
    "A25C_L32G_Y61M_T158F_D204C_F209S_N214M",
    "Y61L_T158V_D204C_F209D_N214V",
    "A25L_R31S_L32G_Y61R_D204C_F209V_N214A",
    "A25M_L32D_Y61C_T158W_D204C_F209E",
    "A25G_L32Q_Y61W_T158S_D204C_F209P_N214K",
    "A25H_L32Q_Y61L_D204C_F209D_N214E",
    "A25K_L32C_Y61M_D204C_F209P_N214E",
    "A25P_L32G_Y61D_D204C_F209V_N214E",
    "A25G_L32H_Y61L_T158V_D204C_F209R_N214P",
    "L32A_Y61G_T158S_D204C_F209Q_N214I",
    "L32M_Y61P_T158F_D204C_F209L_N214E",
    "A25L_L32V_Y61T_T158E_D204C_F209G_N214E",
    "A25R_L32R_Y61R_T158C_F209Q_N214D",
    "A25T_L32M_Y61S_T158V_D204C_F209S_N214T",
    "A25N_L32W_Y61K_D204C_F209P_N214L",
    "A25H_L32M_Y61H_Q143H_T158C_D204C_F209N_N214V",
    "A25F_R31S_L32M_Y61C_T158D_D204C_F209Q_N214D",
    "A25Y_L32A_Y61E_T158Q_D204C_F209L_N214V",
    "L32Q_Y61P_T158Y_D204C_F209V_N214V",
    "A25P_L32G_Y61P_T158H_D204C_F209L_N214K"
]

mutated_sequences = mutations(seq, muts)

df = pd.DataFrame({
    "mut": muts,
    "seq_mutada": mutated_sequences
})

df.to_excel("resultado_seq_mut.xlsx", index=False)


data = pd.read_csv("dataset.csv", sep=";")
print(data.columns.tolist())

# Creamos columna clasificadora: 0 = inactivo, 1 = activo
# Consideramos "activo" si fitness > 0
data['class'] = (data['fitness'] > 0).astype(int)

# # Ver resultado
# print(data.head())

# # Guardamos de nuevo el CSV con la columna añadida
# data.to_csv("dataset_class.csv", index=False)

## Segundo paso: Codificamos clasificadores 

data = pd.read_csv("dataset_class.csv")
seq1 = "wt"
seq2 = "mut"
target_clas = "class"
target_reg = "fitness"


# BLOSUM 

# 1. Concatenar - clasificación
X_b_c_c, y_b_c_c = encode_b_c(data, seq1, seq2, target_clas, basename="data_b_c")
print("BLOSUM concatenar clasificación completado.")

# 2. Diferencia - clasificación
X_b_d_c, y_b_d_c = encode_b_d(data, seq1, seq2, target_clas, basename="data_b_c")
print("BLOSUM diferencia clasificación completado.")

# 3. Transición - clasificación
X_b_t_c, y_b_t_c = encode_b_t(data, seq1, seq2, target_clas, basename="data_b_c")
print("BLOSUM transición clasificación completado.")

# OH

# 4. Concatenar - clasificación
X_oh_c_c, y_oh_c_c = encode_oh_c(data, seq1, seq2, target_clas, basename="data_oh_c")
print("One-Hot concatenar clasificación completado.")

# 5. Diferencia - clasificación
X_oh_d_c, y_oh_d_c = encode_oh_d(data, seq1, seq2, target_clas, basename="data_oh_c")
print("One-Hot diferencia clasificación completado.")

# 6. Transición - clasificación
X_oh_t_c, y_oh_t_c = encode_oh_t(data, seq1, seq2, target_clas, basename="data_oh_c")
print("One-Hot transición clasificación completado.")


# Tercer paso: Entrenamos clasificadores  

# Iniciamos entrenamiento 
dataset_files = [
    "data_oh_c_concat_oh.csv",
    "data_oh_c_dif_oh.csv",
    "data_oh_c_trans_oh.csv",
    "data_b_c_concat_blosum.csv",
    "data_b_c_dif_blosum.csv",
    "data_b_c_trans_blosum.csv"
]

params = param_class
train_function = prepare_classifier

np.random.seed(SEED)
seeds = [np.random.randint(0, 10000) for _ in range(3)]

for dataset in dataset_files:

    print(f"\n Entrenando dataset: {dataset} \n")
    
    # Cargamos dataset
    df = pd.read_csv(dataset)

    # Preparamos los datos
    X_train, X_test, y_train, y_test = prepare_data(
        data=df,
        features=[c for c in df.columns if c not in ['class', 'fitness']],
        target='class',
        scale_data=True
    )

    # Diccionario para este dataset
    train_results_all = {}

    start_total = time()

    # Bucle sobre semillas y modelos
    for seed in seeds:
        for model_cls, params_grid in params.items():
            try:
                train_results = train_model(
                    model=model_cls(),
                    param_grid_to_use=params_grid,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
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

# Cuarto paso: Generamos métricas y gráficas  

pickles = [
    "data_b_c_trans_blosum_train_results.pkl",
    "data_b_c_dif_blosum_train_results.pkl",
    "data_b_c_concat_blosum_train_results.pkl",
    "data_oh_c_trans_oh_train_results.pkl",
    "data_oh_c_dif_oh_train_results.pkl",
    "data_oh_c_concat_oh_train_results.pkl"
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


# =======================================================================================================================================
# YA TENEMOS EL MODELO CLASIFICADOR - ADABOOST 
# Filtramos nuestro dataset donde nos quedamos solo con las muestras con  actividad (==1)
# Tenemos pocas muestras activas y los modelos regresores no funcionan bien
# procedemos a aumentar el dataset haciendo combinaciones de los mutantes 
# =======================================================================================================================================

# DATASET BIDIRECCIONAL 

from itertools import combinations

# Cargamos y filtramos el dataset 
data_f = data[data[target_clas] == 1]
df_f = data_f

# Eliminamos la columna 'class' (no es necesario para el análisis de regresores)
if "class" in df_f.columns:
    df_f = df_f.drop(columns=["class"])

# Seleccionamos la secuencia WT
wt_sequence = df_f["wt"].iloc[0]

# Creamos diccionarios para buscar fitness e id por secuencia mutante
fitness_dict = {}
id_dict = {}

for idx, row in df_f.iterrows():
    mut_seq = row["mut"]
    fitness_dict[mut_seq] = row["fitness"]
    id_dict[mut_seq] = row["id"]

# Añadimos  WT con etiqueta 0
fitness_dict[wt_sequence] = df_f["fitness"].iloc[0]
id_dict[wt_sequence] = 0   

# Crear todos los pares
pairs = []

for m1, m2 in combinations(df_f["mut"], 2):
    pairs.append((m1, m2))
    pairs.append((m2, m1))


for m in df_f["mut"]:
    pairs.append((wt_sequence, m))
    pairs.append((m, wt_sequence))

pairs_df = pd.DataFrame(pairs, columns=["seq_A", "seq_B"])

# Añadimos id de A y B y creamos columna id_pair
pairs_df["id_A"] = pairs_df["seq_A"].map(id_dict)
pairs_df["id_B"] = pairs_df["seq_B"].map(id_dict)
pairs_df["id_pair"] = pairs_df["id_A"].astype(str) + "_" + pairs_df["id_B"].astype(str)

# Añadimos log ratio de fitness A y B
pairs_df["fitness_A"] = pairs_df["seq_A"].map(fitness_dict)
pairs_df["fitness_B"] = pairs_df["seq_B"].map(fitness_dict)
pairs_df["fitness_log_ratio"] = np.log(pairs_df["fitness_A"] / pairs_df["fitness_B"])


print(pairs_df)
print("Total de comparaciones:", len(pairs_df))
pairs_df.to_csv("dataset_regresor.csv")

# ====================================================================================================================================
# Una vez tenemos dataset bidireccional, entrenamos modelos regresores
# ====================================================================================================================================

# Primero: Codificamos regresores

data_f = pd.read_csv("dataset_regresor.csv")
seq1 = "seq_A"
seq2 = "seq_B"
target_reg = "fitness_log_ratio"

# 1. Concatenar - regresión  
X_b_c_rf, y_b_c_rf = encode_b_c(data_f, seq1, seq2, target_reg, basename="data_b_rf")
print("BLOSUM concatenar regresión completado.")

# 2. Diferencia - regresión 
X_b_d_rf, y_b_d_rf = encode_b_d(data_f, seq1, seq2, target_reg, basename="data_b_rf")
print("BLOSUM diferencia regresión completado.")

# 3. Transición - regresión  
X_b_t_rf, y_b_t_rf = encode_b_t(data_f, seq1, seq2, target_reg, basename="data_b_rf")
print("BLOSUM transición regresión completado.")

# 4. Concatenar - regresión 
X_oh_c_rf, y_oh_c_rf = encode_oh_c(data_f, seq1, seq2, target_reg, basename="data_oh_rf")
print("One-Hot concatenar regresión completado.")

# 5. Diferencia - regresión - No hace falta 
X_oh_d_rf, y_oh_d_rf = encode_oh_d(data_f, seq1, seq2, target_reg, basename="data_oh_rf")
print("One-Hot diferencia regresión completado.")

# 6. Transición - regresión - No hace falta 
X_oh_t_rf, y_oh_t_rf = encode_oh_t(data_f, seq1, seq2, target_reg, basename="data_oh_rf")
print("One-Hot transición regresión completado.")


# Segundo: Entrenamos regresores

# Iniciamos entrenamiento 
dataset_files_f = [
    "data_b_rf_concat_blosum.csv",
    "data_b_rf_dif_blosum.csv",
    "data_b_rf_trans_blosum.csv",
    "data_oh_rf_concat_oh.csv",
    "data_oh_rf_dif_oh.csv",
    "data_oh_rf_trans_oh.csv"
]

params = param_regre
train_function = prepare_regressor

np.random.seed(SEED)
seeds = [np.random.randint(0, 10000) for _ in range(3)]

for dataset in dataset_files_f:

    print(f"\n Entrenando dataset: {dataset} \n")
    
    # Cargamos dataset
    df = pd.read_csv(dataset)

    # Preparamos los datos
    X_train, X_test, y_train, y_test = prepare_data(
        data=df,
        features=[c for c in df.columns if c not in ['fitness_log_ratio']],
        target='fitness_log_ratio',
        scale_data=True
    )

    # Diccionario para este dataset
    train_results_all = {}

    start_total = time()

    # Bucle sobre semillas y modelos
    for seed in seeds:
        for model_cls, params_grid in params.items():
            try:
                train_results = train_model(
                    model=model_cls(),
                    param_grid_to_use=params_grid,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
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

 
# Tercero: Métricas y gráficas  

pickles = [
    "data_oh_rf_trans_oh_train_results.pkl",
    "data_oh_rf_dif_oh_train_results.pkl",
    "data_oh_rf_concat_oh_train_results.pkl",
    "data_b_rf_trans_blosum_train_results.pkl",
    "data_b_rf_dif_blosum_train_results.pkl",
    "data_b_rf_concat_blosum_train_results.pkl"
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

# =====================================================================================================================================
# Generamos dataset de control sin duplicación de pares y entrenamos nuevamente los modelos regresores
# =====================================================================================================================================

# Creamos dataset
data_f1 = data[data[target_clas] == 1]
df_f1 = data_f1

# Eliminamos la columna 'class' (no es necesario para el análisis de regresores)
if "class" in df_f1.columns:
    df_f1 = df_f1.drop(columns=["class"])

# Seleccionamos la secuencia WT
wt_sequence1 = df_f1["wt"].iloc[0]

# Creamos diccionarios para buscar fitness e id por secuencia mutante
fitness_dict1 = {}
id_dict1 = {}

for idx, row in df_f1.iterrows():
    mut_seq1 = row["mut"]
    fitness_dict1[mut_seq1] = row["fitness"]
    id_dict1[mut_seq1] = row["id"]

# Añadimos WT con etiqueta 0
fitness_dict1[wt_sequence1] = df_f1["fitness"].iloc[0]
id_dict1[wt_sequence1] = 0   

# Cramos pares sin duplicidad

pairs1 = []

# Pares mutante–mutante (solo una vez)
for m1, m2 in combinations(df_f1["mut"], 2):
    pairs1.append((m1, m2))

# Pares WT–mutante (solo una vez)
for m in df_f1["mut"]:
    pairs1.append((wt_sequence1, m))

pairs_df1 = pd.DataFrame(pairs1, columns=["seq_A", "seq_B"])

# Añadimos IDS y fitness 

pairs_df1["id_A"] = pairs_df1["seq_A"].map(id_dict1)
pairs_df1["id_B"] = pairs_df1["seq_B"].map(id_dict1)

pairs_df1["id_pair"] = (
    pairs_df1["id_A"].astype(str) + "_" + pairs_df1["id_B"].astype(str))

pairs_df1["fitness_A"] = pairs_df1["seq_A"].map(fitness_dict1)
pairs_df1["fitness_B"] = pairs_df1["seq_B"].map(fitness_dict1)

pairs_df1["fitness_log_ratio"] = np.log(
    pairs_df1["fitness_A"] / pairs_df1["fitness_B"])

# Guardamos

print(pairs_df1)
print("Total de comparaciones (sin duplicación):", len(pairs_df1))

pairs_df1.to_csv("dataset_regresor_f.csv", index=False)

# ====================================================================================================================================
# Una vez tenemos dataset bidireccional sin duplicación de pares, entrenamos modelos regresores
# ====================================================================================================================================


# Codificamos regresores

data_f = pd.read_csv("dataset_regresor_f.csv")
seq1 = "seq_A"
seq2 = "seq_B"
target_reg = "fitness_log_ratio"

# 1. Concatenar - regresión  
X_b_c_rf1, y_b_c_rf1 = encode_b_c(data_f, seq1, seq2, target_reg, basename="data_b_rf1")
print("BLOSUM concatenar regresión completado.")

# 2. Diferencia - regresión 
X_b_d_rf1, y_b_d_rf1 = encode_b_d(data_f, seq1, seq2, target_reg, basename="data_b_rf1")
print("BLOSUM diferencia regresión completado.")

# 3. Transición - regresión  
X_b_t_rf1, y_b_t_rf1 = encode_b_t(data_f, seq1, seq2, target_reg, basename="data_b_rf1")
print("BLOSUM transición regresión completado.")

# 4. Concatenar - regresión 
X_oh_c_rf1, y_oh_c_rf1 = encode_oh_c(data_f, seq1, seq2, target_reg, basename="data_oh_rf1")
print("One-Hot concatenar regresión completado.")

# 5. Diferencia - regresión - No hace falta 
X_oh_d_rf1, y_oh_d_rf1 = encode_oh_d(data_f, seq1, seq2, target_reg, basename="data_oh_rf1")
print("One-Hot diferencia regresión completado.")

# 6. Transición - regresión - No hace falta 
X_oh_t_rf1, y_oh_t_rf1 = encode_oh_t(data_f, seq1, seq2, target_reg, basename="data_oh_rf1")
print("One-Hot transición regresión completado.")

# Entrenamos modelos regresores 

# Iniciamos entrenamiento 
dataset_files_f = [
    "data_b_rf1_concat_blosum.csv",
    "data_b_rf1_dif_blosum.csv",
    "data_b_rf1_trans_blosum.csv",
    "data_oh_rf1_concat_oh.csv",
    "data_oh_rf1_dif_oh.csv",
    "data_oh_rf1_trans_oh.csv"
]

params = param_regre
train_function = prepare_regressor

np.random.seed(SEED)
seeds = [np.random.randint(0, 10000) for _ in range(3)]

for dataset in dataset_files_f:

    print(f"\n Entrenando dataset: {dataset} \n")
    
    # Cargamos dataset
    df = pd.read_csv(dataset)

    # Preparamos los datos
    X_train, X_test, y_train, y_test = prepare_data(
        data=df,
        features=[c for c in df.columns if c not in ['fitness_log_ratio']],
        target='fitness_log_ratio',
        scale_data=True
    )

    # Diccionario para este dataset
    train_results_all = {}

    start_total = time()

    # Bucle sobre semillas y modelos
    for seed in seeds:
        for model_cls, params_grid in params.items():
            try:
                train_results = train_model(
                    model=model_cls(),
                    param_grid_to_use=params_grid,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
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


# TERCER PASO: MÉTRICAS Y GRÁFICAS 

pickles = [
    "data_oh_rf1_trans_oh_train_results.pkl",
    "data_oh_rf1_dif_oh_train_results.pkl",
    "data_oh_rf1_concat_oh_train_results.pkl",
    "data_b_rf1_trans_blosum_train_results.pkl",
    "data_b_rf1_dif_blosum_train_results.pkl",
    "data_b_rf1_concat_blosum_train_results.pkl"
    ]

for pkl_file in pickles:
    output_excel = pkl_file.replace("_train_results.pkl", ".xlsx")
    df_cv, df_test = tablas_metricas(pkl_file, output_excel=output_excel)
    print(f"{output_excel} generado")

# Carpeta donde guardar las gráficas
save_dir = "graficas_reg_f1"
os.makedirs(save_dir, exist_ok=True)

for pkl_file in pickles:
    print(f"Generando gráficas para {pkl_file}...")
    graficar_modelos(pkl_file, is_class=False, save_dir=save_dir)
