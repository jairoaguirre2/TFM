import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from modelos import prepare_data, prepare_classifier, prepare_regressor, train_model, evaluated_model
from modelos import param_regre, param_class
from funciones import encode_blosum, encode_blosum_transition 
from funciones import encode_b_c, encode_b_d, encode_b_t, blosum62
from funciones import encode_oh, encode_oh_c, encode_oh_d, encode_oh_t
from time import time
SEED=164

# ======================================================================================================================
# ARCHIVO DE PRUEBA Y DESARROLLO DE LOS MODELOS DE ENTRENAMIENTO 
# En este script se entrenaron modelos preliminares con el dataset publico y se realizaron pruebas experimentales 
# Las funciones aquí utilizadas fueron posteriormente modificadas, optimizadas  y trasladadas a los módulos 
# finales empleados para la obtención de los resultados presentados en el apartado de Resultados.
# Este archivo se incluye con fines de trazabilidad del flujo de trabajo y documentación del proceso de desarrollo,
#  no como código final reproducible
# ====================================================================================================================


# -----------------------------------------------------------------------------------------------------------
# Graficos 
# -----------------------------------------------------------------------------------------------------------



exit()
# -----------------------------------------------------------------------------------------------------------
# Prueba codificación OH modelos regresores 
# -----------------------------------------------------------------------------------------------------------

path = "subset_3_transition_transition_oh.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]


for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })
    
# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_3_t_oh.csv", index=False)
print("Resultados guardados en 'results_train_3_t_oh.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_3_t_oh_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")

exit()


path = "subset_3_diff_diferencia_oh.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

print("Preparando modelos")
# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]


for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })
    
# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_3_d_oh.csv", index=False)
print("Resultados guardados en 'results_train_3_d_oh.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_3_d_oh_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")

exit()

# -----------------------------------------------------------------------------------------------------------
# Prueba codificación OH modelos regresores 
# -----------------------------------------------------------------------------------------------------------

data = pd.read_csv("subset_3.csv")

# Columnas principales del dataset
seq1 = "protSeq1"
seq2 = "protSeq2"
target_col = "tm2"
basename = "subset_3"

# Encoding por concatenación + flatten
X_c, y_c = encode_oh_c(seq1_col=seq1, seq2_col=seq2, target_col=target_col,
    basename=f"{basename}_concat",
    data=data)

print("encode_oh_c completado.")

# Encoding por diferencia entre secuencias (mutante - parental)
X_d, y_d = encode_oh_d(seq1=seq1, seq2=seq2, target_col=target_col,
    basename=f"{basename}_diff",
    data=data)

print("encode_oh_diferencia completado.")

# Encoding tipo transición (posición a posición con One-Hot)
X_t, y_t = encode_oh_t(target_column=target_col, seq1=seq1, seq2=seq2,
    basename=f"{basename}_transition",
    data=data)

print("encode_oh_t completado.")


# -----------------------------------------------------------------------------------------------------------
# Prueba nuevas codificaciones con subset_3
# -----------------------------------------------------------------------------------------------------------

from time import time
SEED=164


path = "subset_3_transition_blosum.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

print("Preparando modelos")
# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

print("Preparando resultados")

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

print("Bucle seed")

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    print("Evaluando modelos")

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    print("Guardando resultados")
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })
    
# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_3_t_blosum.csv", index=False)
print("Resultados guardados en 'results_train_3_t_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_3_t_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")


exit()    

from time import time
SEED=164


path = "subset_3_diferencia_blosum.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

print("Preparando modelos")
# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

print("Preparando resultados")

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

print("Bucle seed")

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    print("Evaluando modelos")

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    print("Guardando resultados")
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })

# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_3_d_blosum.csv", index=False)
print("Resultados guardados en 'results_train_3_d_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_3_d_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")


##################
    


# Columnas principales del dataset
data=pd.read_csv("subset_3.csv")
seq1 = "protSeq1"
seq2 = "protSeq2"
target_col = "tm2"
basename = "subset_3"

# Encoding por concatenación + flatten

X_c, y_c = encode_b_c(seq1=seq1, seq2=seq2, target_col=target_col,
    basename=f"{basename}",
    data=data)

print("encode_b_c completado.")


# Encoding por diferencia entre secuencias (mutante - parental)

X_d, y_d = encode_b_d(seq1=seq1, seq2=seq2, target_col=target_col,
    basename=f"{basename}",
    data=data)
print("encode_b_diferencia completado.")

# Encoding tipo transición (posición a posición con BLOSUM)

X_t, y_t = encode_b_t(target_column=target_col, seq1=seq1, seq2=seq2,
    basename=f"{basename}",
    blosum=blosum62, data=data
)
print("encode_b_t completado.")


# -----------------------------------------------------------------------------------------------------------
# Entrenar modelo encode blosum diff y transition 
# -----------------------------------------------------------------------------------------------------------


from time import time
SEED=164


path = "subset_1_concat_encode_blosum.csv" 
df = pd.read_csv(path)
params = param_regre
train_function = prepare_regressor

print("Preparando modelos")
# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

print("Preparando resultados")

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

print("Bucle seed")

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    print("Evaluando modelos")

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    print("Guardando resultados")
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })

# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_uno_m_blosum.csv", index=False)
print("Resultados guardados en 'results_train_uno_m_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_uno_m_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")

#### da error 



from time import time
SEED=164


path = "subset_1_transition_transition_blosum.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })


# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_trans_blosum.csv", index=False)
print("Resultados guardados en 'results_train_trans_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_trans_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")



 #######################################################


from time import time
SEED=164


path = "subset_1_diff_diferencia_blosum.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })


# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_diff_blosum.csv", index=False)
print("Resultados guardados en 'results_train_diff_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_diff_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")








# -----------------------------------------------------------------------------------------------------------
# Entrenar modelo encode blosum concatenar
# -----------------------------------------------------------------------------------------------------------


from time import time
SEED=164


path = "subset_1_concat_encode_blosum.csv" 
df = pd.read_csv(path)
df = df.sample(200, random_state=SEED)
params = param_regre
train_function = prepare_regressor

# Preparar los datos
X_train, X_test, y_train, y_test = prepare_data(
    path=path,
    features=[col for col in df.columns if col != 'tm2'],
    target='tm2',
    scale_data=True
)

# Preparar modelos regresores
results = []

np.random.seed(SEED)
seeds = [np.random.randint(0,10000) for i in range(len(params))]

for seed, (model, params_grid) in zip(seeds, params.items()): 
    start_total= time()
        
    # Entrenar modelos
    model_trained, y_pred, best_params = train_model(
        model=model(),
        param_grid_to_use=params_grid, 
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    # Evaluar modelos
    model_trained, metrics_df = evaluated_model(model_trained, y_test, y_pred)
    
    # Guardar resultados
    results.append({
        "Model": model.__name__,
        "MSE": metrics_df["MSE"].values[0],
        "MAE": metrics_df["MAE"].values[0],
        "R2": metrics_df["R2"].values[0],
        "Best Params": best_params
    })


# Extraemos resultados
results_df = pd.DataFrame(results)
results_df.to_csv("results_train_uno_m_blosum.csv", index=False)
print("Resultados guardados en 'results_train_uno_m_blosum.csv'")


total_time = (time() - start_total) / 60.0

with open("results_train_uno_m_blosum_time.txt", "w") as f:
    f.write(f"Tiempo total: {total_time:.2f} minutos\n")
    f.write(f"Modelos entrenados: {len(results)}\n")


# -----------------------------------------------------------------------------------------------------------
# Cargar archivos y codificadar de las 3 maneras distintas
# -----------------------------------------------------------------------------------------------------------


# Cargamos el archivo

data = pd.read_csv("train_data.csv")
basename = "info_general_train_data"
output_file = f"{basename}.txt"
print(data)
print(data.head())
print(data.shape)
print(data.dtypes)
print(data.info())
print(data.describe())

# Es un dataset muy grande, vamos a crear distintos subsets de manera aleatoria con los 
# que verificaremos si los modelos funcionan

data["len"] = [len(s) for s in data["protSeq1"]]
data = data[data["len"]<=400]

np.random.seed(164)
states = [np.random.randint(1,1000) for i in range(10)]


subsets = []  
subset_size = 0.001  

for i, s in enumerate(states, start=1):
    subset = data.sample(frac=subset_size, random_state=s)
    subsets.append(subset)
    
    # Guardar cada subset como CSV
    subset_name = f"subset_{i}.csv"
    subset.to_csv(subset_name, index=False)
    print(f"Subset {i} guardado")

print("\nTotal de subsets creados:", len(subsets))



data = pd.read_csv("subset_1.csv")
print(data)
print(data.dtypes)


# PRUEBA DE LAS 3 FUNCIONES DE ENCODING CON subset_1.csv

# Columnas principales del dataset
seq1 = "protSeq1"
seq2 = "protSeq2"
target_col = "tm2"
basename = "subset_1"

# Encoding por concatenación + flatten

X_c, y_c = encode_b_c(seq1=seq1, seq2=seq2, target_col=target_col,
    basename=f"{basename}_concat",
    data=data)

print("encode_b_c completado.")


# Encoding por diferencia entre secuencias (mutante - parental)

X_d, y_d = encode_b_d(seq1=seq1, seq2=seq2, target_col=target_col,
    basename=f"{basename}_diff",
    data=data)
print("encode_b_diferencia completado.")

# Encoding tipo transición (posición a posición con BLOSUM)


X_t, y_t = encode_b_t(target_column=target_col, seq1=seq1, seq2=seq2,
    basename=f"{basename}_transition",
    blosum=blosum62, data=data
)
print("encode_b_t completado.")


