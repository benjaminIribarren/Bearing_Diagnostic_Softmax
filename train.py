import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTRAIN_PATH = os.path.join(BASE_DIR, "dtrain.csv")
PESOS_PATH = os.path.join(BASE_DIR, "pesos.npz")
EPOCHS = 1000
LEARNING_RATE = 0.001
BETA = 0.9
EPSILON = 1e-8
N_CLASES = 4

def cargar_datos(ruta):
    datos = pd.read_csv(ruta, header=None)
    datos = datos.to_numpy(dtype=float)
    X = datos[:, :-4]
    Y = datos[:, -4:]

    return X, Y

def cargar_pesos(ruta):
    pesos = np.load(ruta)
    V = pesos["V"]
    media_Xr = pesos["media_Xr"]
    Vpca = pesos["Vpca"]

    return V, media_Xr, Vpca

def reducir_dimensionalidad(X, V, media_Xr, Vpca):
    Xr = np.tanh(X @ V)
    Xrr = (Xr - media_Xr) @ Vpca

    return Xrr

def normalizar_datos(X):
    media = np.mean(X, axis=0)
    desviacion = np.std(X, axis=0)
    desviacion[desviacion == 0] = 1.0
    X_norm = (X - media) / desviacion

    return X_norm, media, desviacion

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)

    probabilidades = exp_Z / np.sum(
        exp_Z,
        axis=1,
        keepdims=True
    )

    return probabilidades

def entropia_cruzada(Y, probabilidades):
    epsilon = 1e-12
    costo = -np.mean(
        np.sum(
            Y * np.log(probabilidades + epsilon),
            axis=1
        )
    )

    return costo

def entrenar_softmax(X, Y):
    n_muestras, n_features = X.shape
    X_bias = np.hstack((
        X,
        np.ones((n_muestras, 1))
    ))

    W = np.zeros((n_features + 1, N_CLASES))
    acumulador = np.zeros_like(W)

    costos = []

    for epoca in range(EPOCHS):
        Z = X_bias @ W
        probabilidades = softmax(Z)
        costo = entropia_cruzada(Y, probabilidades)
        gradiente = (
            X_bias.T @ (probabilidades - Y)
        ) / n_muestras

        acumulador = (
            BETA * acumulador
            + (1 - BETA) * gradiente**2
        )

        W = W - LEARNING_RATE * gradiente / (
            np.sqrt(acumulador) + EPSILON
        )

        probabilidades = softmax(X_bias @ W)
        costo = entropia_cruzada(Y, probabilidades)
        costos.append(costo)

    return W, costos

if __name__ == "__main__":

    X, Y = cargar_datos(DTRAIN_PATH)
    V, media_Xr, Vpca = cargar_pesos(PESOS_PATH)
    Xrr = reducir_dimensionalidad(
        X, V, media_Xr, Vpca
    )

    X_norm, media_norm, desviacion_norm = normalizar_datos(Xrr)
    W_softmax, costos = entrenar_softmax(X_norm, Y)

    np.savez(
        os.path.join(BASE_DIR, "pesoftmax.npz"),
        W=W_softmax,
        media=media_norm,
        desviacion=desviacion_norm
    )

    pd.DataFrame(costos).to_csv(
        os.path.join(BASE_DIR, "costo.csv"),
        index=False,
        header=False
    )

    plt.figure(figsize=(8, 5))
    plt.plot(costos)
    plt.xlabel("Épocas")
    plt.ylabel("Entropía cruzada")
    plt.title("Convergencia del entrenamiento Softmax")
    plt.grid()
    plt.savefig(
        os.path.join(BASE_DIR, "costo.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()