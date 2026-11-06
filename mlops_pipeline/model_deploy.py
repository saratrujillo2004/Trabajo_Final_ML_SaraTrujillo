"""
Despliegue del modelo mediante una API (FastAPI).

Carga el mejor modelo entrenado y expone el endpoint /predict, que recibe uno o
varios clientes en formato JSON, aplica las mismas reglas
de validación y atributos derivados del entrenamiento, y retorna la probabilidad
de no pago y la predicción de cada cliente.

"""

from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ft_engineering import aplicar_reglas_validacion, crear_atributos_derivados

# Configuración

RUTA_MODELO = Path(__file__).resolve().parent / "resultados" / "mejor_modelo.joblib"
CLASE_NO_PAGO = 0
MAXIMO_REGISTROS = 1000


# Esquema de los datos de entrada

class Cliente(BaseModel):
    """Información de un cliente al momento de solicitar el crédito."""

    tipo_credito: int = Field(examples=[4])
    capital_prestado: float = Field(gt=0, examples=[2000000])
    plazo_meses: int = Field(gt=0, examples=[12])
    edad_cliente: Optional[float] = Field(default=None, examples=[35])
    tipo_laboral: Optional[Literal["Empleado", "Independiente"]] = Field(default=None, examples=["Empleado"])
    salario_cliente: Optional[float] = Field(default=None, examples=[3000000])
    total_otros_prestamos: float = Field(default=0, ge=0, examples=[500000])
    cuota_pactada: float = Field(gt=0, examples=[190000])
    puntaje_datacredito: Optional[float] = Field(default=None, examples=[780])
    cant_creditosvigentes: int = Field(default=0, ge=0, examples=[2])
    huella_consulta: int = Field(default=0, ge=0, examples=[4])
    saldo_mora: Optional[float] = Field(default=None, examples=[0])
    saldo_total: Optional[float] = Field(default=None, examples=[1500000])
    creditos_sectorFinanciero: int = Field(default=0, ge=0, examples=[1])
    creditos_sectorCooperativo: int = Field(default=0, ge=0, examples=[0])
    creditos_sectorReal: int = Field(default=0, ge=0, examples=[1])
    promedio_ingresos_datacredito: Optional[float] = Field(default=None, examples=[1200000])
    tendencia_ingresos: Optional[Literal["Creciente", "Estable", "Decreciente"]] = Field(
        default=None, examples=["Creciente"]
    )


class SolicitudPrediccion(BaseModel):
    """Lote de clientes a evaluar."""

    clientes: list[Cliente] = Field(min_length=1, max_length=MAXIMO_REGISTROS)


class ResultadoCliente(BaseModel):
    """Resultado de la predicción para un cliente."""

    prob_no_pago: float
    prediccion: int
    resultado: str


class RespuestaPrediccion(BaseModel):
    """Resultados del lote completo."""

    total_clientes: int
    clientes_riesgo: int
    resultados: list[ResultadoCliente]

# Lógica de predicción

def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el pipeline entrenado (preprocesador + clasificador)."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecute primero model_training_evaluation.py"
        )
    return joblib.load(ruta)


def predecir(modelo, clientes):
    """Aplica las transformaciones del entrenamiento y retorna las predicciones."""
    df = pd.DataFrame([cliente.model_dump() for cliente in clientes])
    df = aplicar_reglas_validacion(df)
    df = crear_atributos_derivados(df)
    df = df[modelo.feature_names_in_]  # mismas columnas y orden que en el entrenamiento

    posicion = list(modelo.classes_).index(CLASE_NO_PAGO)
    probabilidades = modelo.predict_proba(df)[:, posicion]
    predicciones = modelo.predict(df)

    return [
        ResultadoCliente(
            prob_no_pago=round(float(probabilidad), 4),
            prediccion=int(prediccion),
            resultado="Riesgo de no pago" if prediccion == CLASE_NO_PAGO else "Pago a tiempo",
        )
        for probabilidad, prediccion in zip(probabilidades, predicciones)
    ]


# API

app = FastAPI(
    title="API de riesgo de no pago",
    description="Predice si un cliente pagará a tiempo su crédito.",
    version="1.0.0",
)
modelo_entrenado = cargar_modelo()


@app.get("/")
def estado():
    """Verifica que la API esté funcionando."""
    return {"estado": "activo", "modelo": type(modelo_entrenado[-1]).__name__}


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(solicitud: SolicitudPrediccion):
    """Recibe uno o varios clientes y retorna la predicción de cada uno."""
    try:
        resultados = predecir(modelo_entrenado, solicitud.clientes)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Error al generar la predicción") from error

    return RespuestaPrediccion(
        total_clientes=len(resultados),
        clientes_riesgo=sum(r.prediccion == CLASE_NO_PAGO for r in resultados),
        resultados=resultados,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
