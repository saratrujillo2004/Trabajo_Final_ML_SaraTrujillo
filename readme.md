# Predicción del pago a tiempo de créditos – Proyecto final de Machine Learning

**Estudiante:** Sara Trujillo  
**Curso:** Machine Learning – Universidad Católica Luis Amigó

## 1. Descripción del problema

Una entidad financiera presta dinero a personas y necesita saber qué clientes podrían tener dificultades para pagar sus créditos a tiempo. Contar con esta información le permitiría tomar mejores decisiones al momento de aprobar un crédito y reducir las posibles pérdidas.

Para dar solución a esta necesidad, se desarrolló un proyecto de Machine Learning que busca identificar a los clientes que podrían no pagar a tiempo, utilizando la información disponible sobre ellos. Además, el proyecto incluye diferentes etapas para preparar los datos, entrenar y evaluar varios modelos, revisar si los datos cambian con el tiempo y permitir que el modelo pueda utilizarse mediante una API.

La información utilizada tiene las siguientes características:

- **Base de datos:** contiene 10.763 créditos otorgados entre noviembre de 2024 y abril de 2026, con 23 variables.
- **Variable que se quiere predecir:** `Pago_atiempo`, donde 1 significa que el cliente pagó a tiempo y 0 que no lo hizo.
- **Principal dificultad:** existe una gran diferencia entre los dos grupos, ya que el 95,25 % de los clientes pagó a tiempo y solo el 4,75 % no lo hizo.

Esta diferencia es importante porque el modelo debe aprender a reconocer a los clientes que no pagan, aunque sean una minoría dentro de la base de datos.

## 2. Organización del proyecto

El proyecto está organizado en diferentes archivos, cada uno encargado de una parte del proceso.

```text
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   ├── model_deploy.py
│   └── resultados/
├── Base_de_datos.csv
├── requirements.txt
├── setup.bat
├── .gitignore
└── readme.md
```

La función de cada archivo es la siguiente:

- **`Cargar_datos.ipynb`:** permite cargar la información original y guardarla en un archivo CSV para trabajar con ella.
- **`comprension_eda.ipynb`:** contiene el análisis inicial de los datos, donde se revisan sus características, los errores, los valores faltantes y las diferencias entre los clientes que pagan a tiempo y los que no.
- **`ft_engineering.py`:** aplica las correcciones necesarias, crea nuevas variables y prepara los datos para utilizarlos en los modelos.
- **`model_training_evaluation.py`:** entrena diferentes modelos, compara sus resultados y guarda el que presenta el mejor desempeño según las métricas seleccionadas.
- **`model_monitoring.py`:** revisa si la información cambia con el tiempo y permite consultar los resultados en un tablero de Streamlit.
- **`model_deploy.py`:** contiene el proceso para poner el modelo a disposición de otros sistemas mediante una API.
- **`resultados/`:** almacena los archivos generados durante el proyecto, como el modelo entrenado, las tablas y los gráficos.

El repositorio también cuenta con tres ramas para organizar el desarrollo: `developer`, donde se realizan los cambios; `certification`, donde se prueban; y `main`, que corresponde a la versión de producción.

## 3. Cómo ejecutar el proyecto

Para ejecutar el proyecto, primero se debe clonar el repositorio y abrir una terminal en la carpeta principal.

1. Ejecutar `setup.bat` para crear el entorno virtual e instalar las dependencias necesarias.
2. Activar el entorno virtual con el siguiente comando en PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Ejecutar los archivos en el siguiente orden:

| Proceso | Comando |
|---|---|
| Preparar y transformar los datos | `python mlops_pipeline/ft_engineering.py` |
| Entrenar y comparar los modelos | `python mlops_pipeline/model_training_evaluation.py` |
| Calcular los cambios en los datos | `python mlops_pipeline/model_monitoring.py` |
| Abrir el tablero de monitoreo | `streamlit run mlops_pipeline/model_monitoring.py` |

De esta manera, se pueden ejecutar las diferentes etapas del proyecto desde la preparación de los datos hasta la revisión de los cambios que se presentan con el tiempo.

## 4. Desarrollo del proyecto y principales resultados

### 4.1. Análisis exploratorio de los datos (EDA)

El primer paso fue revisar la base de datos para conocer la información disponible, identificar posibles errores y entender qué diferencias existen entre los clientes que pagan a tiempo y los que no.

Durante este análisis se encontraron los siguientes resultados:

- **Calidad de los datos:** se identificaron valores que no tenían sentido o que estaban fuera de los rangos esperados. Por ejemplo, había edades entre 121 y 123 años, puntajes de Datacrédito fuera de la escala establecida entre 150 y 950, salarios iguales a cero y valores numéricos en la variable `tendencia_ingresos`, que debería contener categorías como creciente, estable o decreciente. Estos valores se trataron como datos faltantes para evitar que afectaran el análisis.

- **Diferencia entre los grupos:** el 95,25 % de los clientes pagó a tiempo, mientras que el 4,75 % no lo hizo. Esta diferencia hace que el problema sea más difícil, porque el modelo tiene muchos más ejemplos de clientes que pagan que de clientes que no pagan. Por eso, no basta con medir cuántas predicciones son correctas en general; también es necesario revisar qué tan bien identifica a los clientes que no pagan.

- **Características de los clientes con mayor riesgo:** se observó que los clientes que no pagan a tiempo tienden a ser más jóvenes, tienen puntajes más bajos en Datacrédito, realizan más consultas en centrales de riesgo, solicitan montos más altos, registran menores ingresos y presentan con mayor frecuencia una tendencia decreciente en sus ingresos. Sin embargo, estas características no permiten identificar por sí solas a todos los clientes que no pagan.

- **Posible fuga de información en `puntaje`:** esta variable separaba perfectamente a los clientes que pagaban a tiempo de los que no. Esto llamó la atención porque una separación tan clara resulta poco habitual en este tipo de problema. Por esa razón, se consideró que podía contener información relacionada con el resultado del pago y se decidió excluirla para evitar que el modelo obtuviera resultados engañosos.

- **Variables que aportaban información similar:** las variables `saldo_total` y `saldo_principal` presentaban una correlación de Spearman de 0,95, lo que indica que estaban muy relacionadas. Se decidió excluir `saldo_principal` y conservar `saldo_total`. También se eliminó `saldo_mora_codeudor`, ya que casi no presentaba variación en sus valores.

- **Datos faltantes:** se encontró que la variable `tendencia_ingresos` tenía más valores faltantes entre los trabajadores independientes. En este grupo faltaba el 41,5 % de la información, frente al 19,7 % de los empleados. Por eso, se decidió tener en cuenta esta situación durante la preparación de los datos.

En general, el EDA permitió conocer mejor la información disponible y tomar decisiones sobre qué corregir, qué variables conservar y cuáles no utilizar en el modelo.

### 4.2. Preparación de los datos e ingeniería de características

Después del análisis exploratorio, se prepararon los datos para que los modelos pudieran trabajar con ellos de forma adecuada.

Las principales acciones fueron las siguientes:

- **Validación de los datos:** se establecieron reglas para identificar valores incorrectos en las variables, tanto en la base inicial como en los datos que se procesen posteriormente.

- **Creación de nuevas variables:** se construyeron características que ayudan a entender mejor la situación financiera de los clientes:
  - `ratio_cuota_salario`: relaciona el valor de la cuota con el salario para conocer qué parte de los ingresos representa el pago del crédito.
  - `ratio_endeudamiento`: relaciona otros préstamos con el salario del cliente para analizar su nivel de endeudamiento.
  - `tiene_mora`: indica si el cliente registra un saldo en mora.
  - `tendencia_sin_info`: permite identificar cuándo no se dispone de información sobre la tendencia de los ingresos.

- **Agrupación de tipos de crédito:** las categorías con muy pocos registros se agruparon en una categoría llamada `Otros`, para evitar trabajar con grupos demasiado pequeños.

- **División de los datos:** la base se dividió en un 80 % para entrenar los modelos y un 20 % para evaluarlos. Se utilizó una división estratificada, lo que significa que se mantuvo aproximadamente la misma proporción de clientes que pagan y que no pagan en ambos grupos.

- **Preparación de las variables:** se construyó un preprocesador con `ColumnTransformer` para aplicar diferentes transformaciones según el tipo de variable:
  - A las variables numéricas con una distribución muy desigual se les completaron los valores faltantes, se les aplicó una transformación logarítmica y se escalaron.
  - A las demás variables numéricas se les completaron los valores faltantes y se escalaron.
  - A las variables categóricas se les completaron los valores faltantes y se convirtieron en columnas numéricas mediante One-Hot Encoding.
  - A las variables ordinales se les asignaron valores numéricos respetando el orden de sus categorías.

Estas transformaciones permiten que los modelos reciban los datos en un formato adecuado para realizar sus predicciones.

### 4.3. Entrenamiento y comparación de los modelos

Una vez preparados los datos, se entrenaron cuatro modelos de clasificación: Random Forest, XGBoost, regresión logística y árbol de decisión.

Como existen muchos más clientes que pagan a tiempo que clientes que no pagan, se utilizó ponderación de clases para dar mayor importancia al grupo minoritario durante el entrenamiento.

Para comparar los modelos se utilizaron varias métricas:

- **Recall:** indica qué proporción de los clientes que realmente no pagaron fue identificada por el modelo.
- **Precision:** muestra qué proporción de los clientes señalados como posibles incumplidores realmente pertenece al grupo que no pagó.
- **F1:** combina las dos métricas anteriores y permite evaluar el equilibrio entre detectar más casos de no pago y evitar falsas alarmas.
- **AUC:** permite evaluar qué tan bien diferencia el modelo entre los clientes que pagan y los que no.
- **Desviación de F1 en validación cruzada:** ayuda a revisar qué tanto cambian los resultados del modelo al evaluarlo en diferentes divisiones de los datos.
- **Tiempo de entrenamiento:** permite comparar cuánto tarda cada modelo en aprender a partir de los datos.

Los resultados obtenidos fueron los siguientes:

| Modelo | Recall | Precision | F1 | AUC | Desv. F1 (CV) | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 0,373 | 0,116 | **0,177** | 0,685 | 0,013 | 0,9 |
| XGBoost | 0,382 | 0,103 | 0,162 | 0,665 | 0,021 | 0,3 |
| Regresión logística | 0,627 | 0,082 | 0,145 | 0,694 | 0,006 | 0,2 |
| Árbol de decisión | 0,559 | 0,059 | 0,106 | 0,572 | 0,008 | 0,2 |

**Modelo seleccionado: Random Forest.**

Se seleccionó Random Forest porque obtuvo el mejor resultado en F1, con un valor de 0,177. Esto significa que, entre los modelos evaluados, fue el que consiguió el mejor equilibrio entre identificar a los clientes que no pagan y evitar falsas alarmas.

También presentó resultados relativamente estables durante la validación cruzada y un tiempo de entrenamiento bajo.

La regresión logística obtuvo el mayor recall, con 0,627. Esto significa que logró identificar una mayor proporción de los clientes que realmente no pagaron. Sin embargo, su precision fue de 0,082, lo que indica que muchas de las personas señaladas por el modelo sí habían pagado a tiempo.

Por otro lado, el AUC de los modelos muestra que todavía existe margen de mejora. En el caso de Random Forest, el resultado fue de 0,685. Esto indica que la capacidad del modelo para diferenciar entre los dos grupos es moderada.

Estos resultados muestran que predecir el incumplimiento sigue siendo un reto. Además, al excluir la variable `puntaje` por la posible fuga de información, el modelo debe basarse en otras características de los clientes para realizar sus predicciones.

### 4.4. Seguimiento de los cambios en los datos (Data Drift)

Después de entrenar el modelo, también es importante revisar si la información de los clientes cambia con el tiempo. Por ejemplo, podrían cambiar los salarios, los montos solicitados, los tipos de crédito o las características de las personas que solicitan préstamos.

Si estos cambios son importantes, el modelo podría dejar de funcionar tan bien como al principio. Para revisar esta situación, se desarrolló un proceso de monitoreo que compara los datos de referencia con los de periodos posteriores.

La población de referencia está formada por los créditos anteriores a abril de 2025. A partir de esta información, se comparan los datos de cada trimestre posterior.

Para realizar las comparaciones se utilizaron diferentes medidas estadísticas:

- **PSI:** ayuda a identificar qué tanto ha cambiado la distribución de una variable.
- **Kolmogorov-Smirnov (KS):** permite comparar las distribuciones de las variables numéricas entre dos periodos.
- **Jensen-Shannon:** mide las diferencias entre dos distribuciones de datos.
- **Chi-cuadrado:** permite revisar si existen cambios en la distribución de las categorías de las variables categóricas.

Los resultados del monitoreo mostraron lo siguiente:

- En todos los trimestres analizados se encontraron entre cinco y seis variables con cambios importantes.
- Los cambios más destacados se presentaron en los montos de los créditos, los salarios, los otros préstamos, las cuotas, los ingresos reportados por Datacrédito, el nivel de endeudamiento y la distribución de los clientes según su tipo de trabajo y de crédito.
- Durante el primer trimestre de 2026 también comenzó a observarse un cambio en las puntuaciones de riesgo generadas por el modelo.

Estos resultados indican que la información de los clientes no se mantiene igual a lo largo del tiempo. Por eso, es importante continuar revisando estos cambios y comprobar si afectan la capacidad del modelo para identificar a los clientes que no pagan a tiempo.

Como recomendación, se propone mantener el monitoreo trimestral y evaluar un posible reentrenamiento del modelo utilizando datos más recientes. Sin embargo, antes de actualizarlo, es necesario comprobar que los cambios observados afecten realmente su desempeño.

El monitoreo se puede consultar en un tablero desarrollado con Streamlit, que cuenta con tres secciones:

- **Resumen ejecutivo:** presenta un resumen de los cambios encontrados, las recomendaciones y una tabla con las variables que requieren atención.

- **Análisis de variables:** permite comparar cómo se distribuye cada variable en el periodo de referencia y en el periodo seleccionado.

- **Evolución temporal y desempeño:** muestra cómo cambian las medidas de PSI a lo largo de los trimestres y permite comparar el porcentaje real de no pago con las puntuaciones de riesgo del modelo.

Este tablero facilita la revisión de los datos y permite identificar qué variables presentan cambios importantes para decidir si es necesario realizar análisis adicionales.

## 5. Conclusión general

El proyecto permitió desarrollar un proceso que abarca desde la revisión y preparación de los datos hasta el entrenamiento, la evaluación y el monitoreo de un modelo de clasificación para identificar posibles casos de no pago.

Durante el análisis exploratorio se encontraron errores, valores faltantes y variables que podían afectar los resultados. Por eso, se aplicaron reglas de validación, se crearon nuevas características y se excluyeron algunas variables que podían generar problemas, como `puntaje`, por la posible fuga de información.

Posteriormente, se compararon cuatro modelos y se seleccionó Random Forest, ya que obtuvo el mejor resultado en la métrica F1. Sin embargo, sus resultados muestran que todavía es difícil identificar correctamente a los clientes que no pagan a tiempo, por lo que sería conveniente seguir evaluando alternativas para mejorar su desempeño.

Finalmente, se incorporó un proceso de monitoreo para revisar cómo cambia la información de los clientes con el tiempo. Esto es importante porque un modelo no solo debe evaluarse cuando se entrena, sino también cuando empieza a utilizarse y recibe información nueva.

En conclusión, el proyecto permitió aplicar los conocimientos de Machine Learning a un problema real de riesgo crediticio y construir una base para continuar mejorando el modelo y evaluar su comportamiento a lo largo del tiempo.