# =====================================================================================================================
# FUNCIONES GENERALES PARA ENCODING DE SECUENCIAS, AUTOENCODER Y MUTACIONES
# =====================================================================================================================

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Align import substitution_matrices
import gzip
import logomaker
import matplotlib.pyplot as plt
from keras import layers, models, losses
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================================================================================
# CONSTANTES
# =====================================================================================================================

aminoacidos = list("ARNDCQEGHILKMFPSTWYV")
PAD_IDX = 20
aa_to_index = {aa: i for i, aa in enumerate(aminoacidos)}
blosum62 = substitution_matrices.load("BLOSUM62")

# =====================================================================================================================
# ENCODING BLOSUM
# =====================================================================================================================

def encode_blosum(seq):
    """
    Codifica una secuencia de aminoácidos usando la matriz BLOSUM62.

    Args:
        seq (str): Secuencia proteica.

    Returns:
        np.ndarray: Matriz de codificación BLOSUM de forma (len(seq), 20).

    Raises:
        ValueError: Si la secuencia está vacía.
    """
    if not seq or len(seq.strip()) == 0:
        raise ValueError("La secuencia está vacía")
    return np.array([blosum62[aa].tolist()[:-4] for aa in seq])


def encode_b_c(data, seq1, seq2, target_col, basename, max_length=400*20*2, blosum=blosum62):
    """
    Codifica dos secuencias concatenadas usando BLOSUM62.

    Args:
        data (pd.DataFrame): DataFrame con secuencias.
        seq1, seq2 (str): Nombres de columnas con secuencias.
        target_col (str): Nombre de columna con la variable objetivo.
        basename (str): Base para guardar CSV.
        max_length (int): Longitud máxima de codificación.
        blosum: Matriz de sustitución BLOSUM.

    Returns:
        X_blosum (np.ndarray): Matriz codificada de forma (num_seqs, max_length).
        y (np.ndarray): Vector de labels.
    """
    X_blosum = np.zeros((len(data), max_length)) 
    for i, (idx, row) in enumerate(data.iterrows()): 
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        blo1 = encode_blosum(seq1_1)
        blo2 = encode_blosum(seq2_2)
        blo_c_f = np.concatenate([blo1, blo2], axis=0).flatten()
        X_blosum[i, :len(blo_c_f)] = blo_c_f
    y = data[target_col].values
    df_blosum = pd.DataFrame(X_blosum)
    df_blosum[target_col] = y
    df_blosum.to_csv(f"{basename}_concat_blosum.csv", index=False)
    return X_blosum, y


def encode_b_fd(data, seq1, seq2, target_col, basename, max_length=400*20*2, blosum=blosum62):
    """
    Codificación BLOSUM62 de la diferencia entre dos secuencias (flatten).

    Returns:
        X_blosum (np.ndarray), y (np.ndarray)
    """
    X_blosum = np.zeros((len(data), max_length)) 
    for i, (idx, row) in enumerate(data.iterrows()): 
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        blo1 = encode_blosum(seq1_1)
        blo2 = encode_blosum(seq2_2)
        blo_fd = blo2.flatten() - blo1.flatten()
        X_blosum[i, :len(blo_fd)] = blo_fd
    y = data[target_col].values
    df_blosum = pd.DataFrame(X_blosum)
    df_blosum[target_col] = y
    df_blosum.to_csv(f"{basename}_flaten_diff_blosum.csv", index=False)
    return X_blosum, y


def encode_b_fdt(data, seq1, seq2, target_col, basename, max_length=6*20+6, blosum=blosum62):
    """
    Codificación BLOSUM62 de la diferencia y transición entre dos secuencias.

    Returns:
        X_blosum (np.ndarray), y (np.ndarray)
    """
    X_blosum = np.zeros((len(data), max_length)) 
    for i, (idx, row) in enumerate(data.iterrows()): 
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        blo1 = encode_blosum(seq1_1)
        blo2 = encode_blosum(seq2_2)
        blo_fd = blo2.flatten() - blo1.flatten()
        blo_transition = encode_blosum_transition(seq1_1, seq2_2, blosum)
        blo_fdt = np.concatenate([blo_fd,blo_transition])
        X_blosum[i, :len(blo_fdt)] = blo_fdt
    y = data[target_col].values
    df_blosum = pd.DataFrame(X_blosum)
    df_blosum[target_col] = y
    df_blosum.to_csv(f"{basename}_flaten_diff_trans_blosum.csv", index=False)
    return X_blosum, y


def encode_b_fdt_s(data, seq1, basename, max_length=6*20+6, blosum=blosum62):
    """
    Codificación BLOSUM62 de una secuencia contra una referencia (sin target).
    Guarda el resultado como archivo .npy.

    Returns:
        X_blosum (np.ndarray)
    """
    X_blosum = np.zeros((len(data), max_length)) 
    for i, (idx, row) in enumerate(data.iterrows()): 
        seq2 = "".join(row.to_list())  
        assert len(seq1) == len(seq2), "Las secuencias deben tener igual longitud"
        blo1 = encode_blosum(seq1)
        blo2 = encode_blosum(seq2)
        blo_fd = blo2.flatten() - blo1.flatten()
        blo_transition = encode_blosum_transition(seq1, seq2, blosum)
        blo_fdt = np.concatenate([blo_fd,blo_transition]).astype(np.float32)
        X_blosum[i, :len(blo_fdt)] = blo_fdt
    np.save(basename, X_blosum)
    return X_blosum


def encode_b_d(data, seq1, seq2, target_col, basename, max_length=400, blosum=blosum62):
    """
    Codificación BLOSUM62 de la diferencia media entre secuencias.

    Returns:
        X_blosum (np.ndarray), y (np.ndarray)
    """
    X_blosum = np.zeros((len(data), max_length)) 
    y = data[target_col].values
    for i, (idx, row) in enumerate(data.iterrows()):
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        blo1 = encode_blosum(seq1_1)
        blo2 = encode_blosum(seq2_2)
        blo_d = np.mean(blo2 - blo1, axis=0)
        X_blosum[i, :len(blo_d)] = blo_d
    df_blosum = pd.DataFrame(X_blosum)
    df_blosum[target_col] = y
    df_blosum.to_csv(f"{basename}_dif_blosum.csv", index=False)
    return X_blosum, y


def encode_blosum_transition(seq1, seq2, blosum=blosum62):
    """
    Calcula la matriz de transición BLOSUM entre dos secuencias.
    
    Returns:
        np.ndarray: Vector de puntuaciones de transición.
    """
    assert len(seq1) == len(seq2), "Las secuencias deben tener igual longitud"
    trans = np.zeros(len(seq1))
    for i, (a1, a2) in enumerate(zip(seq1, seq2)):
        trans[i] = blosum[a1][a2]
    return trans


def encode_b_t(data, seq1, seq2, target_col, basename, max_length=400, blosum=blosum62):
    """
    Codificación de transición BLOSUM62 para dos secuencias.
    
    Returns:
        X_blosum (np.ndarray), y (np.ndarray)
    """
    X_blosum = np.zeros((len(data), max_length))
    y = data[target_col].to_numpy()
    for i, (idx, row) in enumerate(data.iterrows()):
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        blo_transition = encode_blosum_transition(seq1_1, seq2_2, blosum)
        X_blosum[i, :len(blo_transition)] = blo_transition
    df_blosum = pd.DataFrame(X_blosum)
    df_blosum[target_col] = y
    df_blosum.to_csv(f"{basename}_trans_blosum.csv", index=False)
    return X_blosum, y


# =====================================================================================================================
# ENCODING ONE-HOT
# =====================================================================================================================

def encode_oh(seq, dim=20):
    """
    Codifica una secuencia proteica usando one-hot encoding.

    Args:
        seq (str): Secuencia de aminoácidos.
        dim (int): Dimensión del vector one-hot (default=20).

    Returns:
        np.ndarray: Matriz de codificación one-hot de forma (len(seq), dim)
    """
    L = len(seq)
    encoding_oh = np.zeros((L, dim), dtype=np.float32)
    for i, aa in enumerate(seq):
        index = aa_to_index.get(aa)
        if index is not None:
            encoding_oh[i, index] = 1.0
    return encoding_oh


def encode_oh_c(data, seq1, seq2, target_col, basename, max_length=400*20*2):
    """
    Codificación one-hot de dos secuencias concatenadas.
    """
    X_oh = np.zeros((len(data), max_length)) 
    for i, (_, row) in enumerate(data.iterrows()):
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), "Las secuencias deben tener igual longitud"
        oh1 = encode_oh(seq1_1)
        oh2 = encode_oh(seq2_2)
        oh_c_f = np.concatenate([oh1, oh2], axis=0).flatten()
        X_oh[i, :len(oh_c_f)] = oh_c_f
    y = data[target_col].values
    df_oh = pd.DataFrame(X_oh)
    df_oh[target_col] = y
    df_oh.to_csv(f"{basename}_concat_oh.csv", index=False)
    return X_oh, y


def encode_oh_d(data, seq1, seq2, target_col, basename, max_length=400):
    """
    Codificación one-hot de la diferencia media entre secuencias.
    """
    X_oh = np.zeros((len(data), max_length)) 
    y = data[target_col].values
    for i, (idx, row) in enumerate(data.iterrows()):
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), f"Las secuencias deben tener igual longitud"
        oh1 = encode_oh(seq1_1)
        oh2 = encode_oh(seq2_2)
        oh_d = np.mean(oh2 - oh1, axis=0)
        X_oh[i, :len(oh_d)] = oh_d
    X_oh = np.array(X_oh)
    df_oh = pd.DataFrame(X_oh)
    df_oh[target_col] = y
    df_oh.to_csv(f"{basename}_dif_oh.csv", index=False)
    return X_oh, y


def encode_oh_transition(seq1, seq2):
    """
    Calcula transición one-hot entre dos secuencias.
    """
    assert len(seq1) == len(seq2), "Las secuencias deben tener igual longitud"
    trans = np.zeros(len(seq1))
    for i, (a1, a2) in enumerate(zip(seq1, seq2)):
        oh1 = encode_oh(a1)
        oh2 = encode_oh(a2)
        trans[i] = np.sum(oh1 * oh2)
    return trans


def encode_oh_t(data, seq1, seq2,  target_col, basename, max_length=400):
    """
    Codificación de transición one-hot para dos secuencias.
    """
    X_oh = np.zeros((len(data), max_length))
    y = data[target_col].to_numpy()
    for i, (idx, row) in enumerate(data.iterrows()):
        seq1_1, seq2_2 = row[seq1], row[seq2]
        assert len(seq1_1) == len(seq2_2), f"Las secuencias deben tener igual longitud"
        oh_transition = encode_oh_transition(seq1_1, seq2_2)
        l = min(len(oh_transition), max_length)
        X_oh[i, :l] = oh_transition[:l]
    df_oh = pd.DataFrame(X_oh)
    df_oh[target_col] = y
    df_oh.to_csv(f"{basename}_trans_oh.csv", index=False)
    return X_oh, y


# =====================================================================================================================
# TRANSFORMER SKLEARN
# =====================================================================================================================

class SequenceEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer compatible con sklearn para codificación de secuencias
    usando BLOSUM o One-Hot.
    """
    def __init__(self, method='blosum'):
        self.method = method
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoded = []
        for seq in X:
            if self.method == 'blosum':
                vec = encode_blosum(seq).flatten()
            elif self.method == 'onehot':
                vec = encode_oh(seq).flatten()
            else:
                raise ValueError("method debe ser 'blosum' o 'onehot'")
            encoded.append(vec)
        return np.array(encoded)


# =====================================================================================================================
# LOSS CON MASKED Y WEIGHT
# =====================================================================================================================

def masked_categorical_crossentropy(y_true, y_pred):
    """
    Crossentropy categórico ignorando posiciones de padding (mask=0).
    """
    mask = tf.reduce_sum(y_true, axis=-1) != 0.0
    loss = losses.categorical_crossentropy(y_true[mask], y_pred[mask])
    return loss


def masked_weighted_categorical_crossentropy(aa_weights):
    """
    Crossentropy categórico ponderado por peso de aminoácidos, ignorando padding.
    """
    aa_weights = tf.constant(aa_weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        mask = tf.reduce_sum(y_true, axis=-1) != 0.0 
        loss_w = losses.categorical_crossentropy(y_true, y_pred)
        weights_per = tf.reduce_sum(y_true * aa_weights, axis=-1) 
        weighted_loss = loss_w * weights_per
        masked_loss = tf.boolean_mask(weighted_loss, mask)
        return tf.reduce_mean(masked_loss)
    return loss


# =====================================================================================================================
# BUILD AUTOENCODER, ENCODER, DECODER
# =====================================================================================================================

def build_encoder(longitud, canales, latent_dim=64):
    """
    Construye el encoder de un autoencoder 1D.
    """
    encoder_input = layers.Input(shape=(longitud, canales))
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(16, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    shape_before_flatten = x.shape
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim*2, activation='relu')(x)
    bottleneck = layers.Dense(latent_dim, activation='relu')(x)
    encoder = models.Model(encoder_input, bottleneck)
    return encoder, shape_before_flatten


def build_decoder(latent_dim, shape_before_flatten, canales):
    """
    Construye el decoder de un autoencoder 1D.
    """
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(latent_dim*2, activation='relu')(decoder_input)
    x = layers.Dense(shape_before_flatten[1] * shape_before_flatten[2], activation='relu')(x)
    x = layers.Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=7,  activation='relu', padding='same')(x)
    decoder_output = layers.Conv1D(canales, kernel_size=7, activation='softmax', padding='same')(x)
    decoder = models.Model(decoder_input, decoder_output)
    return decoder


def build_autoencoder(longitud, canales, latent_dim=64, learning_rate=0.001, loss=masked_categorical_crossentropy):
    """
    Construye y compila un autoencoder completo 1D.
    """
    encoder, shape_before_flatten = build_encoder(longitud, canales, latent_dim)
    decoder = build_decoder(latent_dim, shape_before_flatten, canales)
    input_layer = layers.Input(shape=(longitud, canales))
    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    autoencoder = models.Model(input_layer, decoded)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder, decoder


# =====================================================================================================================
# LECTURA DE FASTA
# =====================================================================================================================

def read_fasta_gzip(filepath, filetype="fasta", save_uncompressed=True):
    """
    Lee un archivo FASTA comprimido (.gz) y opcionalmente guarda descomprimido.

    Returns:
        list: Lista de objetos SeqIO.
    """
    try:
        with gzip.open(filepath, "rt") as handle:
            sequences = list(SeqIO.parse(handle, filetype))
        print(f"Archivo cargado: {len(sequences)} secuencias leídas.")
        if save_uncompressed:
            output_path = filepath.replace(".gz", "")
            SeqIO.write(sequences, output_path, filetype)
            print(f"Archivo descomprimido guardado en: {output_path}")
        return sequences
    except FileNotFoundError:
        print(f"No se encontró el archivo '{filepath}'.")
        return []
    except Exception as e:
        print(f"Error inesperado: {e}")
        return []


def fasta_to_onehot(fasta_file, max_len):
    """
    Convierte un archivo FASTA en array 3D formato one-hot (20 AA).

    Returns:
        np.ndarray: Shape (num_seqs, max_len, 20)
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    X = np.zeros((len(records), max_len, 20), dtype=np.float32)
    for i, rec in enumerate(records):
        seq = str(rec.seq[:max_len])
        oh = encode_oh(seq)
        X[i, :oh.shape[0], :] = oh
    return X


def fasta_to_onehot_with_pad(fasta_file, max_len):
    """
    Convierte un archivo FASTA en array 3D one-hot con padding (21 dimensiones).

    Returns:
        np.ndarray: Shape (num_seqs, max_len, 21)
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    X = np.zeros((len(records), max_len, 20+1), dtype=np.float32)
    for i, rec in enumerate(records):
        seq = str(rec.seq[:max_len])
        oh = encode_oh(seq, 20+1)
        X[i, :oh.shape[0], :] = oh
        X[i, oh.shape[0]:, PAD_IDX] = 1.0
    return X


# =====================================================================================================================
# OTROS
# =====================================================================================================================

def save_model_summary(model, filename):
    """
    Guarda el summary de un modelo Keras en un archivo de texto.
    """
    with open(filename, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


# =====================================================================================================================
# MUTACIONES
# =====================================================================================================================

def mutation(seq, mutation):
    """
    Aplica una mutación simple a una secuencia.
    """
    seq_list = list(seq)
    for mut in mutation.split("_"):
        pos = int(mut[1:-1])
        new_aa = mut[-1]
        seq_list[pos - 1] = new_aa
    return "".join(seq_list)


def apply_mutations(seq, mutation_string):
    """
    Aplica múltiples mutaciones a una secuencia.
    """
    seq_list = list(seq)
    if not mutation_string or mutation_string.strip() == "":
        return seq
    for mut in mutation_string.split("_"):
        pos = int(mut[1:-1])
        new_aa = mut[-1]
        seq_list[pos - 1] = new_aa
    return "".join(seq_list)


def mutations(seq, mutation_list):
    """
    Genera todas las secuencias mutadas a partir de una lista de mutaciones.
    """
    mutated = []
    for mut in mutation_list:
        mutated.append(apply_mutations(seq, mut))
    return mutated


# =====================================================================================================================
# LOGOS
# =====================================================================================================================

def plot_sequence_logo(sequences, num_positions, title):
    """
    Genera un logo de secuencia con logomaker.

    Args:
        sequences (pd.Series): Secuencias.
        num_positions (int): Número de posiciones a mostrar.
        title (str): Título del gráfico.
    """
    aminoacidos = list('ACDEFGHIKLMNPQRSTVWY')
    freqs = pd.DataFrame(0.0, index=range(num_positions), columns=aminoacidos)
    for i in range(num_positions):
        counts = sequences.str[i].value_counts()
        freqs.loc[i, counts.index] = counts / counts.sum()
    plt.figure(figsize=(10,4))
    logo = logomaker.Logo(
        freqs,
        color_scheme='chemistry',
        stack_order='big_on_top',
        show_spines=False
    )
    logo.ax.set_xticks(range(num_positions))
    logo.ax.set_xticklabels([f'Pos {i+1}' for i in range(num_positions)])
    plt.title(title)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()
