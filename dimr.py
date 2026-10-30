import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTRAIN_PATH = os.path.join(BASE_DIR, "dtrain.csv")
N_COMPONENTES = 20
EPOCHS = 1000
LEARNING_RATE = 0.1 
BETA = 0.9
EPSILON = 1e-8
LAMBDA_RLS = 1.0
DELTA_RLS = 1000.0

def cargar_datos(ruta):

    datos = pd.read_csv(ruta, header=None)

    datos = datos.to_numpy(dtype=float)

    # Separar características y etiquetas
    X = datos[:, :-4]
    Y = datos[:, -4:]

    return X, Y

def estimar_pesos_rls(Xr, X):
    n_muestras, n_componentes = Xr.shape
    n_features = X.shape[1]

    W = np.zeros((n_componentes, n_features))
    P = DELTA_RLS * np.eye(n_componentes)

    for i in range(n_muestras):
        x = Xr[i].reshape(-1, 1)
        y = X[i].reshape(1, -1)
        K = (P @ x) / (LAMBDA_RLS + x.T @ P @ x)
        error = y - x.T @ W
        W = W + K @ error
        P = (
            P - K @ x.T @ P
        ) / LAMBDA_RLS
    return W

def gradiente_no_lineal(X, V, W): # Sin X_bias
    Xr = np.tanh(X @ V)
    error = Xr @ W - X
    costo = 0.5 * np.mean(np.sum(error**2, axis=1))
    gradiente = (X.T @ ((error @ W.T) * (1 - Xr**2))) / X.shape[0]
    return gradiente, costo

def actualizar_rmsprop(V, gradiente, acumulador):

    acumulador = (
        BETA * acumulador
        + (1 - BETA) * gradiente**2
    )

    V = V - LEARNING_RATE * gradiente / (
        np.sqrt(acumulador) + EPSILON
    )

    return V, acumulador

def entrenar_sls(X): # Sin X_bias
    rng = np.random.default_rng(42)
    n_features = X.shape[1]

    V = rng.normal(
        0,
        np.sqrt(1 / n_features),
        size=(n_features, N_COMPONENTES) # Sin el +1 del bias
    )

    acumulador = np.zeros_like(V)
    costos = []

    for epoca in range(EPOCHS):
        Xr = np.tanh(X @ V)
        W = estimar_pesos_rls(Xr, X)

        gradiente, costo = gradiente_no_lineal(X, V, W)

        V, acumulador = actualizar_rmsprop(V, gradiente, acumulador)
        costos.append(costo)

    Xr = np.tanh(X @ V)
    W = estimar_pesos_rls(Xr, X)

    return V, W, costos

def pca_svd(Xr, varianza_objetivo=0.95):
    media = np.mean(Xr, axis=0)
    Xc = Xr - media
    U, S, Vt = np.linalg.svd(
        Xc,
        full_matrices=False
    )
    varianza = S**2

    if np.sum(varianza) == 0:
        raise ValueError("La matriz no presenta variabilidad.")

    varianza_explicada = varianza / np.sum(varianza)
    varianza_acumulada = np.cumsum(varianza_explicada)

    n_componentes = np.searchsorted(
        varianza_acumulada,
        varianza_objetivo
    ) + 1

    Vpca = Vt[:n_componentes, :].T
    Xrr = Xc @ Vpca

    return Xrr, media, Vpca, varianza_acumulada

if __name__ == "__main__":
    X, Y = cargar_datos(DTRAIN_PATH)
    V, W, costos = entrenar_sls(X)

    # Vuelve a ser la proyección original sin el X_bias
    Xr = np.tanh(X @ V) 
    
    Xrr, media_Xr, Vpca, varianza_acumulada = pca_svd(Xr, varianza_objetivo=0.99)

    np.savez(os.path.join(BASE_DIR, "pesos.npz"), V=V, W=W, media_Xr=media_Xr, Vpca=Vpca)

