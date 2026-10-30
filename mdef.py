import numpy as np
import pandas as pd
from scipy.stats import norm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CARPETAS_CLASES = {
    "normal": os.path.join(DATA_DIR, "normal"),
    "externo": os.path.join(DATA_DIR, "fallo", "externo"),
    "interno": os.path.join(DATA_DIR, "fallo", "interno"),
    "rodante": os.path.join(DATA_DIR, "fallo", "rodante")
}

def cargar_csv(ruta):

    datos = pd.read_csv(ruta, header=None)
    X = datos.to_numpy(dtype=float)

    return X

def cargar_clase(carpeta):

    senales = []

    archivos = sorted([
        archivo for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(".csv")
    ])

    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos  en: {carpeta}"
        )

    for archivo in archivos:

        ruta = os.path.join(carpeta, archivo)
        X = cargar_csv(ruta)

        senales.append(X)

    return senales

def ventanas(senal, W=600):

    senal = np.asarray(senal, dtype=float)
    n_muestras, n_canales = senal.shape
    n_ventanas = n_muestras // W
    senal = senal[:n_ventanas * W, :]

    return senal.reshape(n_ventanas, W, n_canales)

def multiescala(senal, tau_max=5):

    senal = np.asarray(senal, dtype=float).ravel()

    escalas = {}

    for tau in range(1, tau_max + 1):
        n_bloques = len(senal) // tau
        datos = senal[:n_bloques * tau]
        senal_escala = datos.reshape(n_bloques, tau).mean(axis=1)
        escalas[tau] = senal_escala

    return escalas

def dispersion_entropy(x, d=3, tau_delay=1, c=3):

    N = len(x)
    M = N - (d - 1) * tau_delay

    if M <= 0:
        return 0.0

    std_x = np.std(x)

    if std_x == 0:
        return 0.0

    # Transformación mediante la distribución normal acumulada
    y = norm.cdf(x, loc=np.mean(x), scale=std_x)

    # Discretización en c categorías
    Y = np.floor(c * y).astype(int) + 1
    Y = np.clip(Y, 1, c)

    # Construcción de patrones
    potencias_c = c ** np.arange(d)
    patrones = []

    for i in range(M):
        Yi = Y[i:i + d * tau_delay:tau_delay]
        k = np.dot(Yi - 1, potencias_c)
        patrones.append(k)

    # Probabilidades de los patrones
    _, counts = np.unique(patrones, return_counts=True)
    p = counts / M

    # Entropía de dispersión normalizada
    DE = -np.sum(p * np.log2(p))
    max_DE = np.log2(c ** d)

    return DE / max_DE if max_DE > 0 else DE

def extraer_features_clase(senales_clase, W=600, tau_max=5, d=3, tau_delay=1, c=3):
    features_clase = []

    for senal in senales_clase:
        ventanas_senal = ventanas(senal, W)

        for ventana in ventanas_senal:

            vector_features = []

            for canal in range(ventana.shape[1]):

                x = ventana[:, canal]
                escalas = multiescala(x, tau_max)

                for escala in range(1, tau_max + 1):

                    y_tau = escalas[escala]
                    de = dispersion_entropy(y_tau, d=d, tau_delay=tau_delay, c=c)
                    vector_features.append(de)

            features_clase.append(vector_features)

    return np.asarray(features_clase, dtype=float).reshape(-1, 4 * tau_max)

if __name__ == "__main__":
 
    senal_normal = cargar_clase(CARPETAS_CLASES["normal"])
    senal_externo = cargar_clase(CARPETAS_CLASES["externo"])
    senal_interno = cargar_clase(CARPETAS_CLASES["interno"])
    senal_rodante = cargar_clase(CARPETAS_CLASES["rodante"])

    features_normal = extraer_features_clase(senal_normal)
    features_externo = extraer_features_clase(senal_externo)
    features_interno = extraer_features_clase(senal_interno)
    features_rodante = extraer_features_clase(senal_rodante)

    y_normal = np.tile([1, 0, 0, 0], (features_normal.shape[0], 1))
    y_externo = np.tile([0, 1, 0, 0], (features_externo.shape[0], 1))
    y_interno = np.tile([0, 0, 1, 0], (features_interno.shape[0], 1))
    y_rodante = np.tile([0, 0, 0, 1], (features_rodante.shape[0], 1))

    dfeatures = np.vstack([
        features_normal,
        features_externo,
        features_interno,
        features_rodante
    ])

    dclass = np.vstack([
        y_normal,
        y_externo,
        y_interno,
        y_rodante
    ])

    rng = np.random.default_rng(42)
    indices = rng.permutation(dfeatures.shape[0])

    dfeatures = dfeatures[indices]
    dclass = dclass[indices]
    
    n_registros = dfeatures.shape[0]
    n_train = int(0.8 * n_registros)

    X_train_original = dfeatures[:n_train]
    X_test_original = dfeatures[n_train:]

    y_train = dclass[:n_train]
    y_test = dclass[n_train:]

    media = np.mean(X_train_original, axis=0)
    desviacion = np.std(X_train_original, axis=0)

    desviacion[desviacion == 0] = 1.0

    X_train = (X_train_original - media) / desviacion
    X_test = (X_test_original - media) / desviacion

    dfeatures = (dfeatures - media) / desviacion

    dtrain = np.hstack((X_train, y_train))
    dtest = np.hstack((X_test, y_test))

    pd.DataFrame(dfeatures).to_csv(
        os.path.join(BASE_DIR, "dfeatures.csv"),
        index=False,
        header=False
    )

    pd.DataFrame(dclass).to_csv(
        os.path.join(BASE_DIR, "dclass.csv"),
        index=False,
        header=False
    )

    pd.DataFrame(dtrain).to_csv(
        os.path.join(BASE_DIR, "dtrain.csv"),
        index=False,
        header=False
    )

    pd.DataFrame(dtest).to_csv(
        os.path.join(BASE_DIR, "dtest.csv"),
        index=False,
        header=False
    )

    print("\nDimensiones finales:")
    print("dfeatures:", dfeatures.shape)
    print("dclass:", dclass.shape)
    print("dtrain:", dtrain.shape)
    print("dtest:", dtest.shape)