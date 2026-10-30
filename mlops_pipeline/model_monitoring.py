"""
Monitoreo del modelo y detección de data drift.

Integra los datos con las predicciones del modelo, divide la información 
en una población de referencia y periodos trimestrales, y calcula métricas 
para identificar posibles cambios en la distribución de las variables (drift), 
utilizando KS, PSI, Jensen-Shannon y chi-cuadrado.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from streamlit import runtime

from ft_engineering import (
    VARIABLES_EXCLUIDAS,
    VARIABLE_OBJETIVO,
    aplicar_reglas_validacion,
    cargar_datos,
    crear_atributos_derivados,
)

# Configuración

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RUTA_MODELO = CARPETA_RESULTADOS / "mejor_modelo.joblib"
RUTA_DRIFT = CARPETA_RESULTADOS / "monitoreo_drift.csv"

# Población de referencia: créditos anteriores a esta fecha
FECHA_CORTE = "2025-04-01"
# Periodos con menos registros que este mínimo no se evalúan
MINIMO_REGISTROS = 100

VARIABLES_NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_total",
    "promedio_ingresos_datacredito",
    "ratio_cuota_salario",
    "ratio_endeudamiento",
    "prob_no_pago", 
]

VARIABLES_CATEGORICAS = ["tipo_laboral", "tipo_credito", "tendencia_ingresos"]

# Umbrales del semáforo
PSI_MODERADO, PSI_FUERTE = 0.10, 0.25
P_VALOR_ALERTA = 0.05


# Datos con pronósticos

def cargar_datos_con_pronosticos():

    """Retorna la base con la probabilidad de no pago estimada por el modelo."""
    df = cargar_datos()
    df = aplicar_reglas_validacion(df)
    df = crear_atributos_derivados(df)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    modelo = joblib.load(RUTA_MODELO)
    X = df.drop(columns=VARIABLES_EXCLUIDAS + [VARIABLE_OBJETIVO])
    posicion = list(modelo.classes_).index(0)
    df["prob_no_pago"] = modelo.predict_proba(X)[:, posicion]

    df["tendencia_ingresos"] = df["tendencia_ingresos"].fillna("Sin información")
    df["periodo"] = df["fecha_prestamo"].dt.to_period("Q").astype(str)
    return df


def separar_poblaciones(df):

    """Separa la población de referencia y los periodos a monitorear."""
    referencia = df[df["fecha_prestamo"] < FECHA_CORTE]
    actual = df[df["fecha_prestamo"] >= FECHA_CORTE]
    return referencia, actual

# Métricas de drift

def calcular_psi(referencia, actual, bins=10):

    """Population Stability Index con cortes en los deciles de la referencia."""
    referencia, actual = referencia.dropna(), actual.dropna()
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf

    prop_ref = np.histogram(referencia, cortes)[0] / len(referencia)
    prop_act = np.histogram(actual, cortes)[0] / len(actual)
    prop_ref = np.clip(prop_ref, 1e-6, None)
    prop_act = np.clip(prop_act, 1e-6, None)
    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))


def calcular_jensen_shannon(referencia, actual, bins=10):

    """Divergencia de Jensen-Shannon (0 = iguales, 1 = totalmente distintas)."""
    referencia, actual = referencia.dropna(), actual.dropna()
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf

    prop_ref = np.histogram(referencia, cortes)[0] / len(referencia)
    prop_act = np.histogram(actual, cortes)[0] / len(actual)
    return float(jensenshannon(prop_ref, prop_act, base=2) ** 2)


def calcular_chi_cuadrado(referencia, actual):

    """P-valor de la prueba chi-cuadrado sobre las frecuencias de cada categoría."""
    tabla = pd.concat(
        [referencia.value_counts(), actual.value_counts()], axis=1
    ).fillna(0)
    return float(chi2_contingency(tabla)[1])


def semaforo_numerica(psi):

    """Asigna el nivel de alerta de una variable numérica según su PSI."""
    if psi >= PSI_FUERTE:
        return "Rojo"
    if psi >= PSI_MODERADO:
        return "Amarillo"
    return "Verde"


def semaforo_categorica(p_valor):

    """Asigna el nivel de alerta de una variable categórica según el chi-cuadrado."""
    if p_valor < P_VALOR_ALERTA / 5:
        return "Rojo"
    if p_valor < P_VALOR_ALERTA:
        return "Amarillo"
    return "Verde"


def calcular_drift(df):

    """Calcula las métricas de drift de cada variable en cada periodo."""
    referencia, actual = separar_poblaciones(df)
    filas = []

    for periodo, datos in actual.groupby("periodo"):
        if len(datos) < MINIMO_REGISTROS:
            continue

        for variable in VARIABLES_NUMERICAS:
            psi = calcular_psi(referencia[variable], datos[variable])
            filas.append({
                "periodo": periodo,
                "registros": len(datos),
                "variable": variable,
                "tipo": "numérica",
                "ks_p_valor": ks_2samp(referencia[variable].dropna(), datos[variable].dropna()).pvalue,
                "psi": psi,
                "jensen_shannon": calcular_jensen_shannon(referencia[variable], datos[variable]),
                "chi2_p_valor": np.nan,
                "alerta": semaforo_numerica(psi),
            })

        for variable in VARIABLES_CATEGORICAS:
            p_valor = calcular_chi_cuadrado(referencia[variable], datos[variable])
            filas.append({
                "periodo": periodo,
                "registros": len(datos),
                "variable": variable,
                "tipo": "categórica",
                "ks_p_valor": np.nan,
                "psi": np.nan,
                "jensen_shannon": np.nan,
                "chi2_p_valor": p_valor,
                "alerta": semaforo_categorica(p_valor),
            })

    return pd.DataFrame(filas)


def resumen_por_periodo(df):

    """Tasa real de no pago frente al pronóstico promedio del modelo, por periodo."""
    _, actual = separar_poblaciones(df)
    resumen = actual.groupby("periodo").agg(
        registros=(VARIABLE_OBJETIVO, "size"),
        no_pago_real=(VARIABLE_OBJETIVO, lambda s: (1 - s.mean()) * 100),
        score_riesgo_medio=("prob_no_pago", lambda s: s.mean() * 100),
    )
    resumen = resumen[resumen["registros"] >= MINIMO_REGISTROS]
    return resumen.round(2)


def generar_recomendaciones(tabla_periodo):

    """Mensajes automáticos según las alertas de un periodo."""
    rojas = tabla_periodo.loc[tabla_periodo["alerta"] == "Rojo", "variable"].tolist()
    amarillas = tabla_periodo.loc[tabla_periodo["alerta"] == "Amarillo", "variable"].tolist()

    mensajes = []
    if "prob_no_pago" in rojas:
        mensajes.append("Los pronósticos del modelo cambiaron fuertemente: se recomienda reentrenar el modelo.")
    if len(rojas) >= 3:
        mensajes.append(f"Hay {len(rojas)} variables con drift fuerte: se recomienda reentrenar el modelo.")
    elif rojas:
        mensajes.append("Revisar el origen del cambio en: " + ", ".join(rojas) + ".")
    if amarillas:
        mensajes.append("Vigilar en el siguiente periodo: " + ", ".join(amarillas) + ".")
    if not mensajes:
        mensajes.append("La población es estable: no se requieren acciones.")
    return mensajes



# Tablero en Streamlit (Rediseñado)

def tablero():
    """Tablero de monitoreo con distribución visual optimizada."""
    st.set_page_config(
        page_title="Monitoreo de Modelo | Data Drift",
        layout="wide"
    )

    # Carga de datos
    df = st.cache_data(cargar_datos_con_pronosticos)()
    drift = st.cache_data(calcular_drift)(df)
    referencia, actual = separar_poblaciones(df)


    # SIDEBAR: Filtros y Configuración General

    st.sidebar.header("Controles del Tablero")
    
    periodos = sorted(drift["periodo"].unique())
    periodo = st.sidebar.selectbox("Seleccionar Periodo Evaluado", periodos, index=len(periodos) - 1)
    tabla_periodo = drift[drift["periodo"] == periodo]

    st.sidebar.divider()
    st.sidebar.caption(f"**Población de Referencia:** < {FECHA_CORTE}")
    st.sidebar.caption(f"**Registros mínimos por periodo:** {MINIMO_REGISTROS}")


    # ENCABEZADO Y KPIS PRINCIPALES

    st.title("Monitoreo del Modelo y Detección de Data Drift")
    st.caption("Evaluación de estabilidad poblacional y desempeño analítico del modelo de pago.")

    # Tarjetas de resumen (Metrics)
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        
        rojos = int((tabla_periodo["alerta"] == "Rojo").sum())
        amarillos = int((tabla_periodo["alerta"] == "Amarillo").sum())
        verdes = int((tabla_periodo["alerta"] == "Verde").sum())
        total_vars = len(tabla_periodo)

        col1.metric("Variables Evaluadas", total_vars, help="Total de variables analizadas en este periodo")
        col2.metric("Drift Fuerte (Rojo)", rojos, delta=f"{rojos} alertas", delta_color="inverse" if rojos > 0 else "normal")
        col3.metric("Drift Moderado (Amarillo)", amarillos, delta=f"{amarillos} avisos", delta_color="off")
        col4.metric("Estables (Verde)", verdes, delta=f"{verdes} óptimas")


    # ESTRUCTURA POR PESTAÑAS (TABS)

    tab_resumen, tab_variables, tab_historico = st.tabs([
        "Resumen Ejecutivo", 
        "Análisis de Variables", 
        "Evolución Temporal y Desempeño"
    ])


    # RESUMEN EJECUTIVO

    with tab_resumen:
        col_rec, col_tabla = st.columns([1, 2], gap="medium")

        with col_rec:
            st.subheader("Recomendaciones del Sistema")
            mensajes = generar_recomendaciones(tabla_periodo)
            for mensaje in mensajes:
                if "reentrenar" in mensaje:
                    st.error(mensaje)
                elif "Revisar" in mensaje or "Vigilar" in mensaje:
                    st.warning(mensaje)
                else:
                    st.success(mensaje)

            st.info(
                f"**Resumen para {periodo}:** Se analiza una cohorte con {tabla_periodo['registros'].iloc[0]} "
                f"registros en relación con el histórico previo a {FECHA_CORTE}.",

            )

        with col_tabla:
            st.subheader(f"Estado de Drift por Variable ({periodo})")
            colores = {
                "Rojo": "background-color: #f8d7da; color: #721c24; font-weight: bold;",
                "Amarillo": "background-color: #fff3cd; color: #856404; font-weight: bold;",
                "Verde": "background-color: #d4edda; color: #155724;"
            }
            st.dataframe(
                tabla_periodo.drop(columns=["periodo"]).style
                .map(lambda valor: colores.get(valor, ""), subset=["alerta"])
                .format(precision=3, na_rep="-"),
                use_container_width=True,
                hide_index=True,
                height=380
            )


    # ANÁLISIS EN DETALLE POR VARIABLE

    with tab_variables:
        st.subheader("Distribución Histórica vs. Periodo Actual")
        
        col_control, col_grafico = st.columns([1, 3], gap="medium")

        with col_control:
            variable = st.selectbox("Seleccionar Variable Numérica", VARIABLES_NUMERICAS)
            
            # Muestra el estado actual de la variable elegida
            alerta_var = tabla_periodo.loc[tabla_periodo["variable"] == variable, "alerta"].values
            psi_var = tabla_periodo.loc[tabla_periodo["variable"] == variable, "psi"].values
            
            if len(alerta_var) > 0:
                st.metric("Estado de Drift", alerta_var[0])
                if not np.isnan(psi_var[0]):
                    st.metric("PSI Calculado", f"{psi_var[0]:.4f}")
            
            st.caption(
                "Comparación de la proporción de clientes acumulados por deciles calculados "
                "sobre la población de referencia."
            )

        with col_grafico:
            datos_periodo = actual[actual["periodo"] == periodo]
            comparacion = pd.concat([
                referencia[variable].rename("Referencia (Base)"),
                datos_periodo[variable].rename(f"Actual ({periodo})"),
            ], axis=1)
            
            cortes = np.unique(np.quantile(referencia[variable].dropna(), np.linspace(0, 1, 11)))
            histograma = pd.DataFrame({
                columna: pd.cut(comparacion[columna], cortes, include_lowest=True)
                .value_counts(normalize=True, sort=False)
                for columna in comparacion.columns
            })
            histograma.index = histograma.index.astype(str)
            st.bar_chart(histograma, stack=False, height=350)


    # EVOLUCIÓN TEMPORAL

    with tab_historico:
        col_linea, col_desempeno = st.columns([3, 2], gap="medium")

        with col_linea:
            st.subheader("Evolución del PSI por Variable")
            evolucion = drift[drift["tipo"] == "numérica"].pivot(
                index="periodo", columns="variable", values="psi"
            )
            
            seleccion = st.multiselect(
                "Variables a graficar:",
                list(evolucion.columns),
                default=["prob_no_pago", "edad_cliente", "capital_prestado"]
            )
            
            if seleccion:
                st.line_chart(evolucion[seleccion], height=320)
            else:
                st.info("Selecciona al menos una variable para desplegar el gráfico.")
                
            st.caption(f"**Límites de Alerta:** PSI ≥ {PSI_MODERADO} (Moderado) | PSI ≥ {PSI_FUERTE} (Fuerte)")

        with col_desempeno:
            st.subheader("No Pago Real vs. Score Medio")
            st.dataframe(
                resumen_por_periodo(df), 
                use_container_width=True,
                height=280
            )
            st.caption(
                "**Nota:** El score no es una probabilidad calibrada directo debido a la "
                "ponderación de clases en el entrenamiento; monitoree la tendencia general en el tiempo."
            )



# Ejecución

def main():
    """Calcula las métricas de drift y las guarda en la carpeta de resultados."""
    df = cargar_datos_con_pronosticos()
    drift = calcular_drift(df)
    drift.to_csv(RUTA_DRIFT, index=False)

    print("Resumen de alertas por periodo:")
    print(pd.crosstab(drift["periodo"], drift["alerta"]))
    print("\nNo pago real vs score de riesgo del modelo (%):")
    print(resumen_por_periodo(df))
    print(f"\nTabla de drift guardada en {RUTA_DRIFT}")


if __name__ == "__main__":
    if runtime.exists():
        tablero()
    else:
        main()