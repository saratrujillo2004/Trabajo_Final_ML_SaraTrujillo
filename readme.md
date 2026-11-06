**Estudiante:** Sara Trujillo  
**Curso:** Machine Learning – Universidad Católica Luis Amigó

# 1. Descripción del problema

Una entidad financiera presta dinero a diferentes personas y necesita saber cuáles de sus clientes podrían tener dificultades para pagar sus créditos a tiempo. Tener esta información le ayudaría a tomar mejores decisiones al momento de aprobar un crédito y evitar posibles pérdidas.

Para trabajar en esta necesidad, se desarrolló un proyecto de Machine Learning que busca identificar a los clientes que podrían no pagar a tiempo, teniendo en cuenta la información disponible de cada uno. El proyecto incluye varias etapas, desde la preparación de los datos y el entrenamiento de diferentes modelos hasta la evaluación de sus resultados, el seguimiento de los cambios en la información y la creación de una API para utilizar el modelo.

La base de datos utilizada tiene las siguientes características:

- **Base de datos:** contiene 10.763 créditos otorgados entre noviembre de 2024 y abril de 2026, con un total de 23 variables.
- **Variable que se quiere predecir:** `Pago_atiempo`, donde 1 significa que el cliente pagó a tiempo y 0 significa que no lo hizo.
- **Principal dificultad:** existe una gran diferencia entre los dos grupos de clientes. El 95,25 % pagó a tiempo y solo el 4,75 % no lo hizo.

Esta diferencia representa una dificultad para el proyecto, porque el modelo debe aprender a reconocer a los clientes que no pagan a tiempo, aunque sean una parte pequeña de la información disponible.

# 2. Organización del proyecto

El proyecto está organizado en diferentes carpetas y archivos. Cada uno tiene una función dentro del proceso, lo que permite mantener el código organizado y trabajar cada etapa por separado.

```text
├── .github/
│   └── workflows/
│       └── sonarcloud.yml
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   ├── model_deploy.py
│   └── resultados/
├── tests/
├── Base_de_datos.csv
├── Dockerfile
├── .dockerignore
├── sonar-project.properties
├── requirements.txt
├── setup.bat
├── .gitignore
└── readme.md
```

A continuación, se explica para qué sirve cada archivo:

- **Cargar_datos.ipynb:** permite cargar la información original y guardarla en un archivo CSV para poder trabajar con ella.
- **comprension_eda.ipynb:** contiene el primer análisis de los datos. Allí se revisan sus características, los posibles errores, los valores faltantes y las diferencias entre los clientes que pagan a tiempo y los que no.
- **ft_engineering.py:** se encarga de corregir algunos datos, crear nuevas variables y preparar la información para que pueda utilizarse en los modelos.
- **model_training_evaluation.py:** permite entrenar diferentes modelos, comparar sus resultados y guardar el que obtiene el mejor resultado según las métricas utilizadas.
- **model_monitoring.py:** permite revisar si los datos van cambiando con el tiempo y consultar los resultados desde un tablero desarrollado en Streamlit.
- **model_deploy.py:** contiene el código necesario para poner el modelo a disposición de otros sistemas mediante una API.
- **resultados/:** guarda los archivos que se generan durante el proyecto, como el modelo entrenado, las tablas y los gráficos.
- **tests/:** contiene las pruebas automáticas que permiten comprobar que las diferentes partes del proyecto funcionan correctamente.
- **Dockerfile y .dockerignore:** permiten preparar todo lo necesario para ejecutar la API dentro de Docker.
- **sonar-project.properties y .github/workflows/sonarcloud.yml:** permiten configurar las pruebas automáticas desde GitHub y revisar la calidad del código con SonarCloud.

El repositorio también cuenta con tres ramas para organizar los cambios del proyecto:

- **developer:** es donde se realizan los cambios y se desarrolla el proyecto.
- **certification:** se utiliza para probar los cambios antes de pasarlos a la versión principal.
- **main:** contiene la versión de producción.

# 3. Cómo ejecutar el proyecto

Para ejecutar el proyecto, primero se debe clonar el repositorio y abrir una terminal desde la carpeta principal.

Después, se deben seguir estos pasos:

1. Ejecutar el archivo `setup.bat`, que se encarga de crear el entorno virtual e instalar las librerías necesarias.
2. Activar el entorno virtual utilizando el siguiente comando en PowerShell:

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
| Encender la API | `python mlops_pipeline/model_deploy.py` |
| Ejecutar las pruebas automáticas | `pytest tests --cov=mlops_pipeline --cov-report=term-missing` |

De esta manera, se pueden ejecutar todas las etapas del proyecto, comenzando por la preparación de los datos y terminando con el monitoreo y el uso del modelo.

## 3.1. Uso de la API

Una vez encendida la API, se debe abrir en el navegador la siguiente dirección:

`http://localhost:8000/docs`

Allí se encuentra una página llamada Swagger, que permite probar la API de una forma sencilla. Para hacerlo, se debe buscar la opción `POST /predict`, seleccionar **Try it out**, ingresar los datos del cliente y finalmente presionar **Execute**.

La API puede recibir uno o varios clientes en una misma solicitud, con un máximo de 1.000 registros. A continuación, se muestra un ejemplo con la información de un cliente:

```json
{
  "clientes": [
    {
      "tipo_credito": 4,
      "capital_prestado": 2000000,
      "plazo_meses": 12,
      "edad_cliente": 35,
      "tipo_laboral": "Empleado",
      "salario_cliente": 3000000,
      "total_otros_prestamos": 500000,
      "cuota_pactada": 190000,
      "puntaje_datacredito": 780,
      "cant_creditosvigentes": 2,
      "huella_consulta": 4,
      "saldo_mora": 0,
      "saldo_total": 1500000,
      "creditos_sectorFinanciero": 1,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 1,
      "promedio_ingresos_datacredito": 1200000,
      "tendencia_ingresos": "Creciente"
    }
  ]
}
```

Después de enviar la información, la API devuelve una respuesta como la siguiente:

```json
{
  "total_clientes": 1,
  "clientes_riesgo": 0,
  "resultados": [
    {
      "prob_no_pago": 0.1954,
      "prediccion": 1,
      "resultado": "Pago a tiempo"
    }
  ]
}
```

En este caso, el modelo indica que la probabilidad de no pago es de 0,1954 y su predicción es 1, lo que significa que clasifica al cliente en el grupo de pago a tiempo.

También se incluyeron validaciones para evitar que se envíen datos incorrectos. Por ejemplo, si se ingresa un plazo negativo o un tipo de trabajo que no está permitido, la API rechaza la solicitud con el código 422 e indica cuál es el error.

## 3.2. Uso con Docker

Otra forma de ejecutar la API es mediante Docker. Esto permite ponerla en funcionamiento sin tener que instalar Python ni todas las librerías directamente en el computador.

Para hacerlo, primero se debe tener Docker Desktop abierto y después ejecutar los siguientes comandos desde la carpeta principal del proyecto:

```powershell
docker build -t api-riesgo-credito .
docker run --rm -p 8000:8000 api-riesgo-credito
```

El primer comando crea la imagen de Docker con los archivos necesarios para ejecutar la API. El segundo inicia el contenedor y permite acceder a ella desde el computador.

Una vez esté funcionando, se puede ingresar nuevamente a la dirección:

`http://localhost:8000/docs`

Allí se puede utilizar la API de la misma manera que se explicó en el punto anterior.

Para detener el contenedor, se debe presionar `Ctrl + C` en la terminal.

# 4. Desarrollo del proyecto y principales resultados

## 4.1. Análisis exploratorio de los datos (EDA)

El primer paso del proyecto fue revisar la base de datos para conocer qué información contenía, identificar posibles errores y entender qué diferencias existían entre los clientes que pagaban a tiempo y los que no.

Durante esta revisión se encontraron varios aspectos importantes:

- **Calidad de los datos:** se encontraron algunos valores que no tenían sentido o que estaban fuera de los rangos esperados. Por ejemplo, había edades entre 121 y 123 años, puntajes de Datacrédito por fuera de la escala establecida entre 150 y 950, salarios iguales a cero y valores numéricos en la variable `tendencia_ingresos`, que debería tener categorías como creciente, estable o decreciente. Estos valores se trataron como datos faltantes para evitar que afectaran los resultados.

- **Diferencia entre los grupos:** el 95,25 % de los clientes pagó a tiempo, mientras que el 4,75 % no lo hizo. Esta diferencia hace que el problema sea más difícil, porque existen muchos más ejemplos de clientes que pagan que de clientes que no pagan. Por esta razón, no es suficiente con revisar cuántas predicciones son correctas en general. También es necesario comprobar qué tan bien identifica el modelo a los clientes que no pagan.

- **Características de los clientes con mayor riesgo:** se observó que los clientes que no pagan a tiempo suelen ser más jóvenes, tienen puntajes más bajos en Datacrédito, realizan más consultas en las centrales de riesgo, solicitan montos más altos, registran menores ingresos y presentan con mayor frecuencia una tendencia decreciente en sus ingresos. Sin embargo, estas características por sí solas no permiten identificar a todos los clientes que no pagan.

- **Posible fuga de información en la variable puntaje:** durante el análisis se encontró que esta variable separaba de manera casi perfecta a los clientes que pagaban a tiempo de los que no. Esto llamó la atención porque una diferencia tan clara no es muy común en este tipo de problemas. Por esa razón, se consideró que podía contener información relacionada con el resultado del pago y se decidió no utilizarla en el modelo. Así se evitaba que el modelo obtuviera resultados engañosos.

- **Variables que contenían información similar:** las variables `saldo_total` y `saldo_principal` tenían una correlación de Spearman de 0,95, lo que significa que estaban muy relacionadas. Por eso, se decidió eliminar `saldo_principal` y conservar `saldo_total`. También se eliminó `saldo_mora_codeudor`, porque casi todos sus valores eran iguales y no aportaba mucha información.

- **Datos faltantes:** se encontró que la variable `tendencia_ingresos` tenía más datos faltantes entre los trabajadores independientes. En este grupo faltaba el 41,5 % de la información, mientras que entre los empleados faltaba el 19,7 %. Por esta razón, se tuvo en cuenta esta diferencia al momento de preparar los datos.

En general, este primer análisis permitió conocer mejor la información disponible y tomar decisiones sobre qué datos era necesario corregir, qué variables se podían conservar y cuáles era mejor no utilizar para entrenar el modelo.

## 4.2. Preparación de los datos e ingeniería de características

Después de revisar la información, el siguiente paso fue preparar los datos para que los modelos pudieran trabajar con ellos correctamente.

Para esto se realizaron varias actividades:

- **Validación de los datos:** se establecieron algunas reglas para identificar valores incorrectos en las variables, tanto en la base de datos inicial como en la información que se procese posteriormente.

- **Creación de nuevas variables:** se crearon algunas características que permiten conocer mejor la situación financiera de cada cliente.

  - `ratio_cuota_salario`: relaciona el valor de la cuota con el salario del cliente. Esto permite conocer qué parte de sus ingresos representa el pago del crédito.
  - `ratio_endeudamiento`: relaciona el valor de otros préstamos con el salario del cliente para tener una idea de su nivel de endeudamiento.
  - `tiene_mora`: indica si el cliente tiene algún saldo en mora.
  - `tendencia_sin_info`: permite identificar los casos en los que no se cuenta con información sobre la tendencia de los ingresos.

- **Agrupación de los tipos de crédito:** algunas categorías tenían muy pocos registros, por lo que se agruparon en una sola categoría llamada `Otros`. Esto permitió evitar que el modelo trabajara con grupos demasiado pequeños.

- **División de los datos:** la base se dividió en dos partes. El 80 % se utilizó para entrenar los modelos y el 20 % para evaluar sus resultados. Esta división se realizó de forma estratificada, lo que significa que se mantuvo aproximadamente la misma proporción de clientes que pagan y que no pagan en ambos grupos.

- **Preparación de las variables:** se utilizó `ColumnTransformer` para aplicar diferentes transformaciones dependiendo del tipo de información.

  - A las variables numéricas que tenían valores muy concentrados o distribuciones desiguales se les completaron los datos faltantes, se les aplicó una transformación logarítmica y después se escalaron.
  - A las demás variables numéricas se les completaron los datos faltantes y se escalaron.
  - A las variables categóricas se les completaron los datos faltantes y se transformaron en columnas numéricas mediante One-Hot Encoding.
  - A las variables ordinales se les asignaron valores numéricos respetando el orden de sus categorías.

Con estos cambios, la información quedó preparada para que los modelos pudieran utilizarla durante el entrenamiento y realizar sus predicciones.

## 4.3. Entrenamiento y comparación de los modelos

Una vez preparados los datos, se entrenaron cuatro modelos de clasificación para comparar sus resultados. Los modelos utilizados fueron Random Forest, XGBoost, regresión logística y árbol de decisión.

Como en la base de datos hay muchos más clientes que pagan a tiempo que clientes que no pagan, se utilizó la ponderación de clases. Esto permite darle mayor importancia al grupo que tiene menos registros durante el entrenamiento.

Para comparar los modelos se utilizaron diferentes métricas:

- **Recall:** muestra qué cantidad de los clientes que realmente no pagaron fue identificada correctamente por el modelo.
- **Precision:** indica cuántos de los clientes que el modelo señaló como posibles incumplidores realmente no pagaron.
- **F1:** combina las dos métricas anteriores y permite revisar el equilibrio entre identificar a los clientes que no pagan y evitar falsas alarmas.
- **AUC:** permite conocer qué tan bien puede diferenciar el modelo entre los clientes que pagan y los que no.
- **Desviación de F1 en validación cruzada:** permite revisar qué tanto cambian los resultados del modelo cuando se evalúa con diferentes divisiones de los datos.
- **Tiempo de entrenamiento:** muestra cuánto tarda cada modelo en aprender a partir de la información.

Los resultados obtenidos fueron los siguientes:

| Modelo | Recall | Precision | F1 | AUC | Desv. F1 (CV) | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0,373 | 0,116 | 0,177 | 0,685 | 0,013 | 0,9 |
| XGBoost | 0,382 | 0,103 | 0,162 | 0,665 | 0,021 | 0,3 |
| Regresión logística | 0,627 | 0,082 | 0,145 | 0,694 | 0,006 | 0,2 |
| Árbol de decisión | 0,559 | 0,059 | 0,106 | 0,572 | 0,008 | 0,2 |

**Modelo seleccionado: Random Forest.**

Se seleccionó Random Forest porque obtuvo el mejor resultado en la métrica F1, con un valor de 0,177. Esto quiere decir que, entre los cuatro modelos evaluados, fue el que logró un mejor equilibrio entre identificar a los clientes que no pagan y evitar falsas alarmas.

Además, presentó resultados relativamente estables durante la validación cruzada y tuvo un tiempo de entrenamiento bajo.

Por otro lado, la regresión logística obtuvo el mayor resultado en Recall, con 0,627. Esto significa que logró identificar una mayor cantidad de los clientes que realmente no pagaron. Sin embargo, su Precision fue de 0,082, lo que indica que muchas de las personas que señaló como posibles incumplidoras sí habían pagado a tiempo.

También es importante mencionar que el AUC de los modelos muestra que todavía hay aspectos por mejorar. En el caso de Random Forest, el resultado fue de 0,685, lo que indica que su capacidad para diferenciar entre los clientes que pagan y los que no es moderada.

Estos resultados muestran que identificar a los clientes que no pagan a tiempo sigue siendo un reto. Además, al excluir la variable `puntaje` por la posible fuga de información, el modelo tuvo que basarse en otras características de los clientes para realizar sus predicciones.

## 4.4. Seguimiento de los cambios en los datos (Data Drift)

Después de entrenar el modelo, también es importante revisar si la información de los clientes cambia con el tiempo. Por ejemplo, pueden cambiar los salarios, los montos de los créditos, los tipos de préstamos o las características de las personas que solicitan estos servicios.

Esto es importante porque, si la información cambia demasiado, el modelo podría dejar de obtener los mismos resultados que tenía cuando fue entrenado.

Para revisar esta situación, se desarrolló un proceso de monitoreo que compara los datos utilizados como referencia con los de los periodos posteriores.

En este caso, se tomó como referencia la información de los créditos anteriores a abril de 2025. A partir de esa información se compararon los datos de cada trimestre posterior.

Para realizar estas comparaciones se utilizaron diferentes medidas:

- **PSI:** permite identificar qué tanto ha cambiado la distribución de una variable entre dos periodos.
- **Kolmogorov-Smirnov (KS):** permite comparar las distribuciones de las variables numéricas entre dos grupos de datos.
- **Jensen-Shannon:** permite medir las diferencias que existen entre dos distribuciones de información.
- **Chi-cuadrado:** permite revisar si existen cambios en la distribución de las categorías de las variables categóricas.

Después de realizar el monitoreo, se encontraron los siguientes resultados:

- En todos los trimestres analizados se encontraron entre cinco y seis variables que presentaban cambios importantes.
- Los cambios más destacados se presentaron en los montos de los créditos, los salarios, los otros préstamos, las cuotas, los ingresos reportados por Datacrédito, el nivel de endeudamiento y la distribución de los clientes según su tipo de trabajo y de crédito.
- Durante el primer trimestre de 2026 también comenzó a observarse un cambio en las puntuaciones de riesgo generadas por el modelo.

Estos resultados muestran que la información de los clientes no se mantiene igual con el paso del tiempo. Por eso, es importante continuar revisando estos cambios y comprobar si pueden afectar los resultados del modelo al momento de identificar a los clientes que no pagan a tiempo.

Como recomendación, se propone continuar realizando el monitoreo cada trimestre y evaluar si es necesario volver a entrenar el modelo con información más reciente. Sin embargo, antes de hacerlo, es importante comprobar que los cambios encontrados realmente estén afectando su desempeño.

Para facilitar esta revisión, se desarrolló un tablero en Streamlit que cuenta con tres secciones:

- **Resumen ejecutivo:** presenta un resumen de los cambios encontrados, las recomendaciones y una tabla con las variables que necesitan mayor atención.
- **Análisis de variables:** permite comparar cómo se comporta cada variable en el periodo de referencia y en el periodo seleccionado.
- **Evolución temporal y desempeño:** muestra cómo cambian los resultados del PSI durante los diferentes trimestres y permite comparar el porcentaje real de no pago con las puntuaciones de riesgo generadas por el modelo.

Este tablero facilita la revisión de la información y permite identificar cuáles son las variables que presentan cambios importantes. De esta manera, se pueden tomar decisiones sobre qué aspectos necesitan una revisión más detallada.

## 4.5. Despliegue del modelo: API y Docker

Para permitir que otros sistemas puedan utilizar el modelo, se desarrolló una API utilizando FastAPI.

Esta API recibe la información de uno o varios clientes, revisa que los datos sean válidos, aplica las mismas transformaciones utilizadas durante el entrenamiento y devuelve los resultados de cada predicción.

Entre los resultados que entrega se encuentra la probabilidad de no pago, la predicción realizada por el modelo y un mensaje que indica si el cliente fue clasificado como alguien que paga a tiempo o no.

Antes de procesar la información, la API verifica que los datos cumplan con las condiciones establecidas. Por ejemplo, revisa que el plazo del crédito sea mayor que cero y que el tipo de trabajo corresponda a una de las opciones permitidas, como "Empleado" o "Independiente".

También permite enviar algunos datos vacíos cuando no se tiene esa información, como la edad o el salario. En estos casos, el modelo completa los valores de la misma manera que se hizo durante el entrenamiento.

Además, la API se preparó para ejecutarse mediante Docker. Para esto se creó una imagen que contiene todo lo necesario para ponerla en funcionamiento, incluyendo Python 3.11, las librerías utilizadas, el código y el modelo entrenado.

Esto permite que la API funcione de la misma manera en diferentes computadores, sin necesidad de instalar cada elemento por separado.

También se tuvieron en cuenta algunos aspectos de seguridad. La imagen incluye únicamente las siete librerías que necesita la API y está configurada para ejecutarse con un usuario que no tiene permisos de administrador.

Finalmente, se realizaron pruebas para comprobar que la API entregara los mismos resultados que el modelo original y que funcionara correctamente tanto dentro como fuera de Docker.

## 4.6. Pruebas automáticas y calidad del código

Para comprobar que las diferentes partes del proyecto funcionaran correctamente, se crearon 31 pruebas automáticas utilizando pytest. Estas pruebas se organizaron en la carpeta `tests/` y se distribuyeron de la siguiente manera:

| Archivo | Qué revisa |
|---|---|
| `test_ft_engineering.py` | Revisa las reglas de validación, las nuevas variables, la división de los datos y el preprocesador. |
| `test_model_training_evaluation.py` | Comprueba el cálculo de las métricas, la configuración de los modelos y el proceso de entrenamiento. |
| `test_model_deploy.py` | Verifica que la API realice correctamente las predicciones y rechace los datos que no cumplen las condiciones. |
| `test_model_monitoring.py` | Revisa las medidas utilizadas para identificar cambios en los datos, el semáforo, las recomendaciones y el tablero. |

Las 31 pruebas se aprobaron y se obtuvo una cobertura del 97 %. Esto significa que las pruebas lograron ejecutar casi todas las partes del código del proyecto.

También se configuró GitHub Actions para que, cada vez que se suban cambios al repositorio, las pruebas se ejecuten automáticamente y se envíe la información a SonarCloud.

SonarCloud es una herramienta que permite revisar la calidad del código, identificar posibles problemas de seguridad, comprobar qué partes del código están cubiertas por las pruebas y encontrar código repetido.

Para realizar esta conexión se utilizó una clave guardada como secreto en GitHub. De esta manera, la clave no queda escrita directamente en los archivos del proyecto.

Los resultados obtenidos en SonarCloud fueron los siguientes:

| Indicador | Resultado |
|---|---:|
| Líneas de código analizadas | 746 |
| Código duplicado | 0,0 % |
| Cobertura de las pruebas | 96,7 % |
| Problemas de fiabilidad y mantenibilidad | 0 |
| Hallazgos de seguridad | 3 |

Durante la revisión se encontraron tres aspectos de seguridad que vale la pena mencionar:

1. **Dirección de inicio de la API:** en el archivo `model_deploy.py`, la API se inicia utilizando la dirección `0.0.0.0`. Esto permite que otros equipos de la misma red puedan intentar acceder a ella. Para limitar el acceso únicamente al computador donde se ejecuta, se podría utilizar la dirección `127.0.0.1`.

2. **Instalación de librerías en Docker:** el Dockerfile permite instalar librerías que pueden ejecutar código durante su instalación. Para evitar este comportamiento, se podría agregar la opción `--only-binary :all:`, que permite instalar únicamente paquetes que ya vienen preparados.

3. **Versiones de las librerías:** el Dockerfile no tiene definidas las versiones exactas de todas las librerías secundarias que utiliza el proyecto. Para mejorar este aspecto, se podría agregar un archivo que registre las versiones exactas y sus respectivas huellas digitales o hashes. Esto ayudaría a comprobar que las librerías instaladas sean las esperadas.

Se decidió dejar estos hallazgos documentados y aceptar los riesgos por tratarse de un proyecto académico que se ejecuta en un computador personal y no utiliza información real de clientes.

Sin embargo, si el proyecto se llevara a un entorno de producción, sería necesario aplicar las correcciones mencionadas para mejorar su seguridad.

# 5. Conclusión general

El desarrollo de este proyecto permitió trabajar en las diferentes etapas de un proceso de Machine Learning, comenzando por la revisión y preparación de los datos, continuando con el entrenamiento y la evaluación de los modelos y terminando con el monitoreo de la información a lo largo del tiempo.

Durante el análisis inicial se encontraron varios problemas en la base de datos, como valores incorrectos, información faltante y variables que podían afectar los resultados. Por esta razón, fue necesario establecer algunas reglas para corregir los datos, crear nuevas características y eliminar variables que podían generar problemas, como `puntaje`, debido a la posible fuga de información.

Después de preparar la información, se entrenaron y compararon cuatro modelos. Finalmente, se seleccionó Random Forest porque obtuvo el mejor resultado en la métrica F1. Sin embargo, los resultados también muestran que todavía es difícil identificar correctamente a los clientes que no pagan a tiempo. Por eso, sería importante continuar probando otras alternativas y buscar formas de mejorar el modelo.

Otro aspecto importante fue el desarrollo del proceso de monitoreo, que permitió identificar que la información de los clientes presenta cambios con el paso del tiempo. Esto demuestra que no es suficiente con evaluar un modelo únicamente cuando se entrena. También es necesario revisar cómo se comporta cuando recibe información nueva y comprobar si sigue realizando buenas predicciones.

Por otra parte, se desarrolló una API para que el modelo pudiera ser utilizado por otros sistemas y se preparó su ejecución mediante Docker. También se crearon pruebas automáticas y se configuró SonarCloud para revisar la calidad del código. Como resultado, se obtuvo una cobertura del 96,7 %, no se encontró código duplicado y se identificaron tres aspectos de seguridad que quedaron documentados.

En general, este proyecto permitió aplicar los conocimientos aprendidos durante el curso de Machine Learning a un problema relacionado con el riesgo crediticio. Además, permitió conocer otras actividades que hacen parte del desarrollo de un modelo, como la preparación de los datos, las pruebas, el monitoreo y la forma de ponerlo en funcionamiento para que pueda ser utilizado por otros sistemas.