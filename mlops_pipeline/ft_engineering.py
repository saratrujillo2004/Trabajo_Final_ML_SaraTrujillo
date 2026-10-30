
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Configuración

RUTA_DATOS = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"

VARIABLE_OBJETIVO = "Pago_atiempo"

# Variables que no se usan como  variables predictoras (esto se ve conclusiones del EDA)

VARIABLES_EXCLUIDAS = [
    "puntaje",              # posible fuga de información
    "saldo_principal",      # es redundante con saldo_total (Spearman 0,95)
    "saldo_mora_codeudor",  # tiene valores constantes (hay solo 3 valores distintos de 0)
    "fecha_prestamo",       # se conserva solo para el monitoreo
]

# Variables numéricas con una distribución muy desigual: Se les aplicará una 
# transformación logarítmica para reducir la asimetría y facilitar 
# el trabajo del modelo con estos datos.

NUMERICAS_ASIMETRICAS = [
    "capital_prestado",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "saldo_mora",
    "saldo_total",
    "promedio_ingresos_datacredito",
    "ratio_cuota_salario",
    "ratio_endeudamiento",
]

#Variables numéricas sin una asimetría marcada:
# únicamente se completarán los valores faltantes y se aplicará un escalado
# para que las variables tengan rangos comparables.

NUMERICAS = [
    "plazo_meses",
    "edad_cliente",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "tiene_mora",
    "tendencia_sin_info",
]

CATEGORICAS = ["tipo_laboral", "tipo_credito"]

ORDINALES = ["tendencia_ingresos"]
ORDEN_TENDENCIA = ["Decreciente", "Estable", "Creciente"]


# Carga

def cargar_datos(ruta=RUTA_DATOS):
    
    """Va a lee la base de datos en CSV."""
    return pd.read_csv(ruta)


# Reglas de validación

def aplicar_reglas_validacion(df):

    """Convierte en NaN los valores que incumplen las reglas definidas en el EDA."""
    df = df.copy()

    df.loc[~df["edad_cliente"].between(18, 100), "edad_cliente"] = np.nan

    df.loc[~df["puntaje_datacredito"].between(150, 950), "puntaje_datacredito"] = np.nan

    df.loc[df["salario_cliente"] <= 0, "salario_cliente"] = np.nan

    df.loc[~df["tipo_laboral"].isin(["Empleado", "Independiente"]), "tipo_laboral"] = np.nan

    df.loc[~df["tendencia_ingresos"].isin(ORDEN_TENDENCIA), "tendencia_ingresos"] = np.nan

    # Solo se conservan registros con una variable objetivo válida (0 o 1)
    df = df[df[VARIABLE_OBJETIVO].isin([0, 1])]

    return df


# Atributos derivados

def crear_atributos_derivados(df):

    """Crea las variables nuevas propuestas en el EDA"""
    df = df.copy()

    df["ratio_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"]
    df["ratio_endeudamiento"] = df["total_otros_prestamos"] / df["salario_cliente"]
    df["tiene_mora"] = (df["saldo_mora"] > 0).astype(int)
    df["tendencia_sin_info"] = df["tendencia_ingresos"].isna().astype(int)

    # Agrupa los tipos de crédito poco frecuentes (6, 7, 68) en "Otros"
    df["tipo_credito"] = df["tipo_credito"].where(
        df["tipo_credito"].isin([4, 9, 10]), "Otros"
    ).astype(str)

    return df
