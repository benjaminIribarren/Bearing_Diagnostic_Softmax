import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DTEST_PATH = os.path.join(BASE_DIR, "dtest.csv")
PESOS_PATH = os.path.join(BASE_DIR, "pesos.npz")
PESOFTMAX_PATH = os.path.join(BASE_DIR, "pesoftmax.npz")

N_CLASES = 4

NOMBRES_CLASES = [
    "Normal",
    "Externo",
    "Interno",
    "Rodante"
]

def cargar_datos(ruta):

    datos = pd.read_csv(ruta, header=None)
    datos = datos.to_numpy(dtype=float)

    X = datos[:, :-4]
    Y = datos[:, -4:]

    return X, Y

def cargar_pesos_reduccion(ruta):

    pesos = np.load(ruta)

    V = pesos["V"]
    media_Xr = pesos["media_Xr"]
    Vpca = pesos["Vpca"]

    return V, media_Xr, Vpca


def cargar_pesos_softmax(ruta):

    pesos = np.load(ruta)

    W = pesos["W"]
    media = pesos["media"]
    desviacion = pesos["desviacion"]

    return W, media, desviacion

def reducir_dimensionalidad(X, V, media_Xr, Vpca):
    # Proyección pura sin bias artificial en la capa oculta
    Xr = np.tanh(X @ V)
    Xrr = (Xr - media_Xr) @ Vpca
    return Xrr

def normalizar_datos(X, media, desviacion):

    X_norm = (X - media) / desviacion

    return X_norm

def softmax(Z):

    Z = Z - np.max(Z, axis=1, keepdims=True)

    exp_Z = np.exp(Z)

    probabilidades = exp_Z / np.sum(
        exp_Z,
        axis=1,
        keepdims=True
    )

    return probabilidades

def predecir(X, W):

    n_muestras = X.shape[0]

    X_bias = np.hstack((
        X,
        np.ones((n_muestras, 1))
    ))

    probabilidades = softmax(X_bias @ W)

    predicciones = np.argmax(
        probabilidades,
        axis=1
    )

    return probabilidades, predicciones

def confusion_mtx(y_real, y_pred):

    cm = np.zeros(
        (N_CLASES, N_CLASES),
        dtype=int
    )

    np.add.at(cm, (y_real, y_pred), 1)

    return cm

def metricas(cm):

    resultados = []

    for i in range(N_CLASES):

        TP = cm[i, i]

        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision = (
            TP / (TP + FP)
            if TP + FP > 0 else 0
        )

        recall = (
            TP / (TP + FN)
            if TP + FN > 0 else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0
        )

        resultados.append([
            precision,
            recall,
            f1
        ])

    return np.array(resultados)

if __name__ == "__main__":
    X, Y = cargar_datos(DTEST_PATH)

    V, media_Xr, Vpca = cargar_pesos_reduccion(
        PESOS_PATH
    )

    W, media, desviacion = cargar_pesos_softmax(
        PESOFTMAX_PATH
    )
    Xrr = reducir_dimensionalidad(
        X, V, media_Xr, Vpca
    )

    X_norm = normalizar_datos(
        Xrr, media, desviacion
    )

    probabilidades, y_pred = predecir(
        X_norm, W
    )

    y_real = np.argmax(Y, axis=1)

    cm = confusion_mtx(y_real, y_pred)

    resultados = metricas(cm)
    # Guardar F-scores por clase
    df_fscores = pd.DataFrame({
        "Clase": NOMBRES_CLASES,
        "F-score": resultados[:, 2]
    })

    fscore_macro = np.mean(resultados[:, 2])

    df_fscores.loc[len(df_fscores)] = [
        "Promedio macro",
        fscore_macro
    ]

    df_fscores.to_csv(
        os.path.join(BASE_DIR, "fscores.csv"),
        index=False
    )

    pd.DataFrame(cm).to_csv(
        os.path.join(BASE_DIR, "cmatriz.csv"),
        index=False,
        header=False
    )

    df_metricas = pd.DataFrame(
        resultados,
        columns=["Precision", "Recall", "F-score"],
        index=NOMBRES_CLASES
    )

    df_metricas.to_csv(
        os.path.join(BASE_DIR, "metricas.csv")
    )

    df_predicciones = pd.DataFrame({
        "Clase_real": y_real,
        "Clase_predicha": y_pred
    })

    for i, nombre in enumerate(NOMBRES_CLASES):
        df_predicciones[f"P_{nombre}"] = probabilidades[:, i]

    df_predicciones.to_csv(
        os.path.join(BASE_DIR, "predicciones.csv"),
        index=False
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de confusión")
    plt.colorbar()
    posiciones = np.arange(N_CLASES)
    plt.xticks(posiciones, NOMBRES_CLASES, rotation=45)
    plt.yticks(posiciones, NOMBRES_CLASES)

    for i in range(N_CLASES):
        for j in range(N_CLASES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",color="black")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()

    plt.savefig(
        os.path.join(BASE_DIR, "matriz.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


