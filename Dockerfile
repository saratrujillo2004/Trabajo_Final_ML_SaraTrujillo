# Imagen base: Python 3.11 en versión liviana
FROM python:3.11-slim

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Dependencias: solo las que necesita la API, tomadas de requirements.txt
COPY requirements.txt .
RUN grep -E "^(pandas|numpy|scipy|scikit-learn|joblib|fastapi|uvicorn)==" requirements.txt > requirements_api.txt \
    && pip install --no-cache-dir -r requirements_api.txt

# Código de la API y modelo entrenado
COPY mlops_pipeline/ft_engineering.py mlops_pipeline/model_deploy.py ./mlops_pipeline/
COPY mlops_pipeline/resultados/mejor_modelo.joblib ./mlops_pipeline/resultados/

# Usuario sin privilegios de administrador (buena práctica de seguridad)
RUN useradd --create-home appuser
USER appuser

# Puerto en el que escucha la API
EXPOSE 8000

# Comando que enciende la API al iniciar el contenedor
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "mlops_pipeline"]