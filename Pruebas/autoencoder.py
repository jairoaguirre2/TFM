import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import load_model

from funciones import (read_fasta_gzip, fasta_to_onehot, build_autoencoder, build_encoder, build_decoder,
                       save_model_summary, masked_categorical_crossentropy, masked_weighted_categorical_crossentropy)

# ==================================================================================================================================
# Exploración de Autoencoder para aprendizaje no supervisado de secuencias de hidrolasas
# - Clusterización al 30% de identidad (MMseqs2) para evitar redundancia
# - Separación train/valid por clusters
# - Codificación one-hot con padding y masking
# - Autoencoder con pérdida categórica ponderada por frecuencia de aminoácidos
# ================================================================================================================================

# Lectura del FASTA original

seqs_autoencoder = read_fasta_gzip("EC_3_reviewed_200_400.fasta.gz", save_uncompressed=True)
print(f"Total de secuencias: {len(seqs_autoencoder)}")
lengths = [len(s.seq) for s in seqs_autoencoder]
print(f"Longitud media: {sum(lengths)/len(lengths):.2f}")

plt.hist(lengths, bins=40, color="purple")
plt.xlabel("Longitud (aa)")
plt.ylabel("Número de secuencias")
plt.title("Distribución de longitudes de secuencia")
plt.savefig("Histograma lon 200-400.png", dpi=300)
plt.close()

# Lectura del archivo TSV de MMseqs2 realizado en WSL

file_path_clusters30_tsv = "EC_3_reviewed_200_400_clusters30.tsv"
df = pd.read_csv(file_path_clusters30_tsv, sep="\t", header=None, names=["representative", "member"])
print(f"\nArchivo leído: {len(df)} secuencias asignadas a clusters")

num_clusters = df["representative"].nunique()
print(f"Número total de clusters: {num_clusters}")

cluster_sizes = df.groupby("representative").size().sort_values(ascending=False)
print("\nClusters más grandes:")
print(cluster_sizes.head())

print("\nEstadísticas de tamaño de cluster:")
print(cluster_sizes.describe())

plt.figure(figsize=(8, 5))
plt.hist(cluster_sizes, bins=50)
plt.title("Distribución del tamaño de clusters (30% identidad)")
plt.xlabel("Número de secuencias por cluster")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("Cluster 30% similitud.png", dpi=300)
plt.close()


# Selección de clusters de validación (~5,000 secuencias)

fasta_file = "EC_3_reviewed_200_400.fasta"
seq_valid = 5000
random_seed = 42

clusters = list(cluster_sizes.index)
random.seed(random_seed)
random.shuffle(clusters)

valid_clusters = []
train_clusters = []
valid_count = 0

for rep in clusters:
    size = cluster_sizes[rep]
    if valid_count < seq_valid:
        valid_clusters.append(rep)
        valid_count += size
    else:
        train_clusters.append(rep)

print(f"Clusters de validación: {len(valid_clusters)} ({valid_count} secuencias aprox.)")
print(f"Clusters de entrenamiento: {len(train_clusters)}")

with open("valid_clusters.txt", "w") as f:
    f.write("\n".join(valid_clusters))
with open("train_clusters.txt", "w") as f:
    f.write("\n".join(train_clusters))


# Limpieza de IDs y creación de FASTA finales
# Pasamos de "sp|A0A009IHW8|ABTIR_ACIB9" a  "A0A009IHW8"

def clean_id(seq_id):
    if seq_id.startswith("sp|"):
        return seq_id.split("|")[1]
    return seq_id

valid_set = set(df[df["representative"].isin(valid_clusters)]["member"])
train_set = set(df[df["representative"].isin(train_clusters)]["member"])

train_records, valid_records = [], []

for record in SeqIO.parse(fasta_file, "fasta"):
    clean = clean_id(record.id)
    record.id = clean
    record.description = ""
    if clean in valid_set:
        valid_records.append(record)
    elif clean in train_set:
        train_records.append(record)

SeqIO.write(train_records, "train_sequences.fasta", "fasta")
SeqIO.write(valid_records, "valid_sequences.fasta", "fasta")

# Guardamos las métricas 
# sys.stdout = open("resumen_dataset.txt", "w", encoding="utf-8") 

# ===========================================================================
# Entrenamiento del Autoencoder para dataset de hidrolasas al 30% de identidad
# ===========================================================================s

# Parámetros de entrenamiento
basename = "prueba_mask_weight"
max_len = 400     
train_fasta = "train_sequences.fasta"
valid_fasta = "valid_sequences.fasta"
early_stop = EarlyStopping(
    monitor='val_loss',      
    patience=10,             
    restore_best_weights=True)

# Convertimos el archivo fasta a one-hot

X_train = fasta_to_onehot(train_fasta, max_len)
X_valid = fasta_to_onehot(valid_fasta, max_len)
print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_valid.shape}")

print(X_train.dtype, X_valid.dtype)
print("Min:", X_train.min(), "Max:", X_train.max())
print("Min:", X_valid.min(), "Max:", X_valid.max())

# Calculmos la frecuencia de cada aminoácido
aa_freq = np.sum(X_train, axis=(0, 1))
aa_freq = aa_freq / np.sum(aa_freq)

# Calculamos pesos inversos a la frecuencia
aa_weights = 1.0 / (aa_freq + 1e-8)
aa_weights = aa_weights / np.mean(aa_weights)

print("Pesos por aminoácido:", aa_weights)

loss_wm = masked_weighted_categorical_crossentropy(aa_weights)


# Construmos el Autoencoder

autoencoder, encoder, decoder = build_autoencoder(max_len, 20, loss=loss_wm)

save_model_summary(autoencoder, f"autoencoder_{basename}_summary.txt")
save_model_summary(encoder, f"encoder_{basename}_summary.txt")
save_model_summary(decoder, f"decoder_{basename}_summary.txt")

print("Resúmenes de modelos guardados en archivos .txt")

# Entrenamos autoencoder

history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_valid, X_valid),
    epochs=50,
    batch_size=128,
    shuffle=True,
    callbacks=[early_stop])

# Guardamos métricas 

df_history = pd.DataFrame({
    "epoch": range(1, len(history.history['loss']) + 1),
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss']
})
df_history.to_csv(f"history_{basename}.csv", index=False)


# Ploteamos métricas

plt.plot(df_history['epoch'], df_history['loss'], label='train_loss')
plt.plot(df_history['epoch'], df_history['val_loss'], label='val_loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"loss_curves_{basename}.png", dpi=300)
plt.show()

# Guardamos los modelos

autoencoder.save(f"autoencoder_{basename}.h5")
encoder.save(f"encoder_{basename}.h5")
decoder.save(f"decoder_{basename}.h5")

# ==========================================================================================================================
# Reconstruimos el output
# ==========================================================================================================================

aa_list=list("ARNDCQEGHILKMFPSTWYV")
basename = "reconstruccion_w_labels_validation"
# Datos 
max_len = 400
X_test = fasta_to_onehot("valid_sequences.fasta", max_len)
#X_test = fasta_to_onehot("train_sequences.fasta", max_len)[:200]
print("Shape X_test:", X_test.shape)

# Calcular la frecuencia de cada aminoácido
aa_freq = np.sum(X_test, axis=(0, 1))
aa_freq = aa_freq / np.sum(aa_freq)

# Calcular pesos inversos a la frecuencia
aa_weights = 1.0 / (aa_freq + 1e-8)
aa_weights = aa_weights / np.mean(aa_weights)
loss_wm = masked_weighted_categorical_crossentropy(aa_weights)

# Modelos
autoencoder = load_model(
    "autoencoder_prueba_mask_weight.h5",
    compile=False
)
encoder = load_model("encoder_prueba_mask_weight.h5")
decoder = load_model("decoder_prueba_mask_weight.h5")

# Reconstrucción 
X_reconstructed = autoencoder.predict(X_test)

# Identidad
matches = np.argmax(X_test, axis=2) == np.argmax(X_reconstructed, axis=2)
accuracy_per_seq = matches.mean(axis=1)
print("Reconstrucción media:", accuracy_per_seq.mean())

# Decodificación OH - seq 
def decode_onehot(array):
    indices = np.argmax(array, axis=1)
    return ''.join(aa_list[i] for i in indices if i < len(aa_list))

# Ejemplo
i = 0 
print("Original:     ", decode_onehot(X_test[i])[:100])
print("Reconstruida: ", decode_onehot(X_reconstructed[i])[:100])

print(np.sum(X_test[i], axis=1))
print(np.round(np.sum(X_reconstructed[i], axis=1)))

test = X_test[i]
reconstructed = X_reconstructed[i]
mask = tf.reduce_sum(test, axis=-1) != 0.0
print(mask)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
sns.heatmap(test[mask].T, cmap="Blues", cbar=True)
plt.title("Original")
plt.xlabel("Posición en la secuencia")
plt.ylabel("Canal (aa)")
plt.yticks(np.arange(len(aa_list))+0.5, aa_list)
decoded_seq = decode_onehot(test[mask])
seq_len = len(decoded_seq)
plt.xticks(
    ticks=np.arange(0, seq_len, 5) + 0.5,
    labels=[decoded_seq[i] for i in range(0, seq_len, 5)]
)

plt.subplot(2,1,2)
sns.heatmap(reconstructed[mask].T, cmap="Blues", cbar=True)
plt.title("Reconstruida")
plt.xlabel("Posición en la secuencia")
plt.ylabel("Canal (aa)")
plt.yticks(np.arange(len(aa_list))+0.5, aa_list)
decoded_seq = decode_onehot(reconstructed[mask])
seq_len = len(decoded_seq)
plt.xticks(
    ticks=np.arange(0, seq_len, 5) + 0.5,
    labels=[decoded_seq[i] for i in range(0, seq_len, 5)]
)

plt.tight_layout()
plt.savefig(f"loss_curves_{basename}.png", dpi=300)



