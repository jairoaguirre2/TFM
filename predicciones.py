from funciones import encode_b_fdt_s
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm 
from modelos import prepare_data
from sklearn.preprocessing import StandardScaler
import itertools

# =======================================================
# Inferencia masiva sobre librería combinatoria de mutantes
# 1. Generación de la librería combinatoria (20^6 variantes)
# 2. Codificación FDT (Flatten–Difference–Transition) basada en BLOSUM
# 3. Escalado de features usando parámetros de entrenamiento
# 4. Predicción mediante ensamblado de modelos (clasificación y regresión)
# 5. Almacenamiento de resultados para análisis posterior
# =======================================================


# Generamos la librería combinatoria (20^6 variantes posibles)

aminoacidos=list("ARNDCQEGHILKMFPSTWYV")
aa_wt = "ANPYERGPNPTDALLEASSGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSLAWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNHMINRASSTVRSRIDSSRLAVMGHSMGGGGTLRLASQRPDLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYVELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ"
positions = (24, 31, 60, 157, 208, 213)
seq1 = "".join(aa_wt[p] for p in positions)  # WT en posiciones hotspot
print(seq1)

df = pd.DataFrame(
    itertools.product(aminoacidos, repeat=6),
    columns=["pos1", "pos2", "pos3", "pos4", "pos5", "pos6"]
)

df.to_csv("combinatorial_library.csv", index=False)

df =  pd.read_csv("combinatorial_library.csv")
print(df.shape)


# Cargamos datasets de entrenamiento para ajustar escaladores
# (necesarios para aplicar la misma normalización en inferencia)

clss = pd.read_csv("classifier_M_flatten_diff_trans_blosum.csv")
reg = pd.read_csv("regressor_M_flatten_diff_trans_blosum.csv")

features_reg = [c for c in reg.columns if c not in ['log_fitness']]
X = reg[features_reg].to_numpy()
scaler_reg = StandardScaler()
X = scaler_reg.fit_transform(X)

features_clss = [c for c in clss.columns if c not in ['class']]
X = clss[features_clss].to_numpy()
scaler_clss = StandardScaler()
X = scaler_clss.fit_transform(X)

# Cargamos la librería combinatoria y modelos preentrenados, entrenamos en bloques para evitar problemas de memoria
df = pd.read_csv("combinatorial_library.csv")
n = len(df)
# n = 64000000
chunks = np.arange(0, n+1, 6400)

with open("models_adam_for_inference_hotspots.pkl", "rb") as f:
    models_classi = pickle.load(f)

with open("models_grad_regressor_for_inference_hotspots.pkl", "rb") as f:
    models_regres = pickle.load(f)

predictions_classi = []
predictions_regres = []

for i in tqdm(range(len(chunks)-1), total=len(chunks)-1, desc="Classi&Regress"):

    X = encode_b_fdt_s(df.iloc[chunks[i]:chunks[i+1]], seq1, basename="cod_combinatorial_library_b_fdt.npy")
    X_scaled = scaler_clss.transform(X)
    preds_models = np.zeros((len(models_classi), X.shape[0]))
    for j, m in enumerate(models_classi):
        est = m["estimator"]
        preds_models[j, :] = est.predict_proba(X_scaled)[:,1]

    mean_pred = preds_models.mean(axis=0)
    predictions_classi.append(mean_pred)

    X_scaled = scaler_reg.transform(X)
    preds_models = np.zeros((len(models_regres), X.shape[0]))
    for j, m in enumerate(models_regres):
        est = m["estimator"]
        preds_models[j, :] = est.predict(X_scaled)

    mean_pred = preds_models.mean(axis=0)
    predictions_regres.append(mean_pred)

# Guardamos resultados
predictions_classi = np.concatenate(predictions_classi).astype(np.float32)
print(predictions_classi.shape)
np.save("models_adam_for_inference_hotspots_predictions.npy", predictions_classi)

predictions_regres = np.concatenate(predictions_regres).astype(np.float32)
print(predictions_regres.shape)
np.save("models_grad_regressor_for_inference_hotspots_predictions.npy", predictions_regres)


df["Class_proba"] = predictions_classi
df["Regress_pred"] = predictions_regres

df.to_csv("combinatorial_library_with_predictions_total.csv")

# ==========================================================================================================
# Análisis de la librería combinatoria
# 1. Analizar la distribución de las predicciones (clasificación y regresión)
# 2. Filtrar y rankear mutantes prometedores
# 3. Evaluar diversidad de secuencias seleccionadas
# 4. Explorar el espacio mutacional mediante reducción de dimensionalidad
# 5. Seleccionar candidatos finales para validación experimental
# ==========================================================================================================

import matplotlib.pyplot as plt
import logomaker
from funciones import plot_sequence_logo
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargamos archivo
df = pd.read_csv("combinatorial_library_with_predictions_total.csv")

# Examinamos y realizamos estadísticas y gráficos
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df["Class_proba"].describe(percentiles=[0.9, 0.99, 0.999]))
print(df["Regress_pred"].describe(percentiles=[0.9, 0.99, 0.999]))
print(df[["Class_proba", "Regress_pred"]].corr())

plt.figure()
plt.hist(df["Class_proba"], bins=100)
plt.axvline(0.5)
plt.xlabel("Class_proba")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades del modelo clasificador")
plt.savefig("distribucion_class.png", dpi=300)
plt.close()



plt.figure()
plt.hist(df["Regress_pred"], bins=100)
plt.axvline(0)
plt.xlabel("Regress_pred")
plt.ylabel("Frecuencia")
plt.title("Distribución de predicciones del modelo regresor")
plt.savefig("distribucion_reg.png", dpi=300)
plt.close()


# Filtramos mutantes >0.5 tanto en algoritmo clasificador como regresor y nos quedamos con el top 0.001%

df_pre = df[(df["Class_proba"] > 0.5) & (df["Regress_pred"] > 0.5)].copy()
df_pre["score"] = df_pre["Class_proba"] * df_pre["Regress_pred"]


filtro = df_pre["score"].quantile(0.999)
df_top = df_pre[df_pre["score"] >= filtro]

df_top = df_top.sort_values("score", ascending=False)
print(df_top)
df_top.to_csv("dataset_filtrado_rankeado.csv", index=False)


# Estudiamos diversidad y distribución

df_ranked = pd.read_csv("dataset_filtrado_rankeado.csv")


plot_sequence_logo(df_ranked[[f'pos{i+1}' for i in range(6)]].astype(str).agg(''.join, axis=1),
                   num_positions=6,
                   title="Sequence logo - Dataset filtrado rankeado")


# Reconstruimos las secuencias mutantes completas (mut_SM) y se codifican
# usando el mismo esquema FDT empleado durante el entrenamiento de los modelos

df_ranked["mut_SM"] = (
    df_ranked["pos1"] +
    df_ranked["pos2"] +
    df_ranked["pos3"] +
    df_ranked["pos4"] +
    df_ranked["pos5"] +
    df_ranked["pos6"]
)

df_ranked["wt_SM"] = "ALYTFN" 
print(df_ranked[["wt_SM","mut_SM"]].head())

df_ranked.to_csv("dataset_ranked_reconstruido.csv", index=False)

positions = ['pos1','pos2','pos3','pos4','pos5','pos6']
X_input = df_ranked[positions].copy()
wt_SM = "ALYTFN"   

X_fdt = encode_b_fdt_s(
    data=X_input,
    seq1=wt_SM,
    basename="ranked_dataset",
    max_length=6*20+6
)
 
X_wt = encode_b_fdt_s(
    data=pd.DataFrame([list(wt_SM)], columns=['pos1','pos2','pos3','pos4','pos5','pos6']),
    seq1=wt_SM,
    basename="WT",
    max_length=6*20+6
)

print(X_fdt.shape)
print(X_wt.shape)

# Selección del número óptimo de clusters mediante Silhouette score y BIC

sil_scores = []
bic_scores = []
K = range(2, 15)  

for k in K:
    kmeans = GaussianMixture(n_components=k, random_state=42)
    labels = kmeans.fit_predict(X_fdt)

    print(len(np.unique(labels)))
    score = silhouette_score(X_fdt, labels)

    bic = kmeans.bic(X_fdt)
    sil_scores.append(score)
    bic_scores.append(bic)

    print(f"K={k}, Silhouette={score:.4f}, BIC={bic:.1f}")
  

sil_scores = np.array(sil_scores)
bic_scores = np.array(bic_scores)

plt.figure(figsize=(6,4))
plt.plot(K, (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min()), marker='o', label="SIL")
plt.plot(K, (bic_scores - bic_scores.min()) / (bic_scores.max() - bic_scores.min()), marker='o', label="BIC")
plt.xlabel("Número de componentes (k)")
plt.ylabel("Score")
plt.legend()
plt.title("Selección del número óptimo de clusters")
plt.tight_layout()
plt.savefig("Sil+Bic.png", dpi=300)
plt.close()


# Exploración del espacio mutacional mediante PCA y t-SNE

pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.concatenate([X_fdt, X_wt], axis=0))

wt_pca = X_pca[-1:]
X_pca = X_pca[:-1]

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

df_pca["score"] = df_ranked["score"].values

# Gráfico PCA por score
plt.figure(figsize=(8,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["score"], cmap="viridis", s=10, alpha=0.7)
plt.scatter(wt_pca[0,0], wt_pca[0,1], c="red", marker="*", s=200, label="WT")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (FDT encoding) – Dataset rankeado")
plt.colorbar(label="Score")
plt.legend()
plt.tight_layout()
plt.savefig("PCA.png", dpi=300)
plt.close()


X_all = np.concatenate([X_fdt, X_wt], axis=0)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_all)

wt_tsne = X_tsne[-1:]  
X_tsne = X_tsne[:-1]

df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
df_tsne["score"] = df_ranked["score"].values

# Gráfico t-SNE
plt.figure(figsize=(8,6))
plt.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"], c=df_tsne["score"], cmap="viridis", s=10, alpha=0.7)
plt.scatter(wt_tsne[0,0], wt_tsne[0,1], c="red", marker="*", s=200, label="WT")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.title("t-SNE (FDT encoding) – Dataset rankeado")
plt.colorbar(label="Score")
plt.legend()
plt.tight_layout()
plt.savefig("tSNE.png", dpi=300)
plt.close()


# Clustering del espacio mutacional 

# Gráfico PCA con clusters
k = 8
df_vis = pd.DataFrame(X_pca, columns=["PC1","PC2"])
df_vis["score"] = df_ranked["score"].values

kmeans = GaussianMixture(n_components=k, random_state=42)
clusters = kmeans.fit_predict(X_fdt)
df_vis["cluster"] = clusters


plt.figure(figsize=(10,6))

colors = plt.cm.tab10(np.linspace(0,1,k))
for c in range(k):
    cluster_points = df_vis[df_vis["cluster"]==c]
    plt.scatter(cluster_points["PC1"], cluster_points["PC2"], 
                s=15, alpha=0.7, color=colors[c], label=f"Cluster {c}")


plt.scatter(wt_pca[0,0], wt_pca[0,1], c="red", marker="*", s=250, label="WT")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA + KMeans Clustering (k={k})")
plt.legend()
plt.tight_layout()
plt.savefig("PCA+clust.png", dpi=300)
plt.close()

# Gráfico t-SNE con clusters
df_vis_tsne = pd.DataFrame(X_tsne, columns=["TSNE1","TSNE2"])
df_vis_tsne["score"] = df_ranked["score"].values
df_vis_tsne["cluster"] = clusters  # reutilizamos los clusters que ya calculaste

plt.figure(figsize=(10,6))
k = len(np.unique(clusters))
colors = plt.cm.tab10(np.linspace(0,1,k))

for c in range(k):
    cluster_points = df_vis_tsne[df_vis_tsne["cluster"]==c]
    plt.scatter(cluster_points["TSNE1"], cluster_points["TSNE2"], 
                s=15, alpha=0.7, color=colors[c], label=f"Cluster {c}")

plt.scatter(wt_tsne[0,0], wt_tsne[0,1], c="red", marker="*", s=250, label="WT")

plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.title(f"t-SNE + Clusters (k={k})")
plt.legend()
plt.tight_layout()
plt.savefig("tSNE+clusters.png", dpi=300)
plt.close()

df_top = pd.read_csv("dataset_ranked_reconstruido.csv")
df_top = df_top.sort_values("score", ascending=False)

# Seleccionamos los mejores mutantes

df_top["cluster"] = df_vis.loc[df_top.index, "cluster"].values

# Elegimos top 5 de cada cluster 
top_per_cluster = df_top.groupby("cluster").head(5).sort_values(by="cluster")

print(top_per_cluster)

top_final = top_per_cluster[top_per_cluster["mut_SM"] != top_per_cluster["wt_SM"]]

# Nos quedamos con el mejor mutante por cluster

top_final.to_csv("mutantes_seleccionados.csv", index=False)
top_final_best = top_final.groupby("cluster").head(1).sort_values(by="cluster")
top_final_best.to_csv("mutantes_seleccionados_best.csv", index=False)

plot_sequence_logo(top_final['mut_SM'].astype(str),
                   num_positions=6,
                   title="Sequence logo - Mutantes_seleccionados")

plot_sequence_logo(top_final_best['mut_SM'].astype(str),
                   num_positions=6,
                   title="Sequence logo - Mutantes_seleccionados best")
