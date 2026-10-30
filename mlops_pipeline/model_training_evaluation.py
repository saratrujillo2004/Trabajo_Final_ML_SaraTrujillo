"""
Entrenamiento y evaluación de modelos.

Entrena varios modelos de clasificación utilizando el preprocesador
de ft_engineering.py, compara su rendimiento mediante métricas 
para el desbalance de clases y guarda el modelo que obtenga los mejores resultados.

"""

import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ft_engineering import preparar_datos

matplotlib.use("Agg")  # guarda los gráficos en archivos sin abrir ventanas

# Configuración

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RUTA_MODELO = CARPETA_RESULTADOS / "mejor_modelo.joblib"

# Clase de interés: los clientes que NO pagan a tiempo 
CLASE_INTERES = 0
SEMILLA = 42


# Métricas

def summarize_classification(y_real, y_pred, y_proba):
    """Calcula las métricas de evaluación tomando como clase de interés el no pago."""
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, pos_label=CLASE_INTERES, zero_division=0),
        "recall": recall_score(y_real, y_pred, pos_label=CLASE_INTERES),
        "f1": f1_score(y_real, y_pred, pos_label=CLASE_INTERES),
        "auc": roc_auc_score(y_real == CLASE_INTERES, y_proba),
    }

# Construcción, entrenamiento y evaluación de un modelo

def build_model(nombre, clasificador, preprocesador, X_train, y_train, X_test, y_test):

    """Entrena un modelo dentro de un pipeline y retorna el modelo y sus resultados."""
    modelo = Pipeline([
        ("preprocesador", preprocesador),
        ("clasificador", clasificador),
    ])

    # Consistency: F1 en validación cruzada estratificada de 5 partes
    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    f1_no_pago = make_scorer(f1_score, pos_label=CLASE_INTERES)
    f1_cv = cross_val_score(modelo, X_train, y_train, cv=validacion, scoring=f1_no_pago)

    # Scalability: tiempo de entrenamiento
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    # Performance: métricas sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)
    posicion = list(modelo.classes_).index(CLASE_INTERES)
    y_proba = modelo.predict_proba(X_test)[:, posicion]

    resultados = {"modelo": nombre}
    resultados.update(summarize_classification(y_test, y_pred, y_proba))
    resultados["f1_cv_media"] = f1_cv.mean()
    resultados["f1_cv_desv"] = f1_cv.std()
    resultados["tiempo_seg"] = tiempo

    return modelo, resultados

# Modelos a comparar

def definir_modelos(y_train):
    """Define los modelos de clasificación y 
        configura la ponderación de clases en cada uno para tener en 
        cuenta el desbalance de los datos."""
    proporcion = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "Regresión logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMILLA
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=SEMILLA
        ),
        "Random forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, min_samples_leaf=20,
            n_jobs=-1, random_state=SEMILLA
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=proporcion, n_estimators=300, max_depth=4,
            learning_rate=0.05, eval_metric="logloss", random_state=SEMILLA
        ),
    }

# Gráficos comparativos

def graficar_metricas(tabla):
    """Gráfico de barras con las métricas principales de cada modelo."""
    ax = tabla.set_index("modelo")[["recall", "precision", "f1", "auc"]].plot(
        kind="bar", figsize=(10, 5), rot=0
    )
    ax.set_title("Comparación de modelos (clase de interés: no paga a tiempo)")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "comparacion_metricas.png")
    plt.close()


def graficar_curvas_roc(modelos, X_test, y_test):

    """Curvas ROC de todos los modelos en un mismo gráfico."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo in modelos.items():
        RocCurveDisplay.from_estimator(
            modelo, X_test, y_test, pos_label=CLASE_INTERES, name=nombre, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_title("Curvas ROC (clase de interés: no paga a tiempo)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "curvas_roc.png")
    plt.close()


# Flujo completo
def main():
    
    """Entrena, compara, selecciona y guarda el mejor modelo."""
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()

    modelos_entrenados = {}
    lista_resultados = []
    for nombre, clasificador in definir_modelos(y_train).items():
        print(f"Entrenando: {nombre}...")
        modelo, resultados = build_model(
            nombre, clasificador, preprocesador, X_train, y_train, X_test, y_test
        )
        modelos_entrenados[nombre] = modelo
        lista_resultados.append(resultados)

    tabla = pd.DataFrame(lista_resultados).sort_values("f1", ascending=False)
    tabla.to_csv(CARPETA_RESULTADOS / "tabla_resumen.csv", index=False)

    print("\nTabla resumen:")
    print(tabla.round(3).to_string(index=False))

    graficar_metricas(tabla)
    graficar_curvas_roc(modelos_entrenados, X_test, y_test)

    mejor = tabla.iloc[0]["modelo"]
    joblib.dump(modelos_entrenados[mejor], RUTA_MODELO)
    print(f"\nMejor modelo: {mejor}. Guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
