"""Pruebas de la API de predicción."""

import pytest
from fastapi.testclient import TestClient

from model_deploy import app, cargar_modelo

cliente_api = TestClient(app)

CLIENTE = {
    "tipo_credito": 4, "capital_prestado": 2000000, "plazo_meses": 12, "edad_cliente": 35,
    "tipo_laboral": "Empleado", "salario_cliente": 3000000, "total_otros_prestamos": 500000,
    "cuota_pactada": 190000, "puntaje_datacredito": 780, "cant_creditosvigentes": 2,
    "huella_consulta": 4, "saldo_mora": 0, "saldo_total": 1500000, "creditos_sectorFinanciero": 1,
    "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
    "promedio_ingresos_datacredito": 1200000, "tendencia_ingresos": "Creciente",
}


def test_estado_de_la_api():
    respuesta = cliente_api.get("/")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "activo"


def test_prediccion_de_un_cliente():
    respuesta = cliente_api.post("/predict", json={"clientes": [CLIENTE]})
    assert respuesta.status_code == 200
    resultado = respuesta.json()["resultados"][0]
    assert 0 <= resultado["prob_no_pago"] <= 1
    assert resultado["prediccion"] in (0, 1)


def test_prediccion_por_lotes_con_datos_faltantes():
    incompleto = {"tipo_credito": 9, "capital_prestado": 5000000, "plazo_meses": 6, "cuota_pactada": 900000}
    respuesta = cliente_api.post("/predict", json={"clientes": [CLIENTE, incompleto]})
    assert respuesta.status_code == 200
    assert respuesta.json()["total_clientes"] == 2


@pytest.mark.parametrize("cambio", [{"tipo_laboral": "Pensionado"}, {"plazo_meses": -3}])
def test_datos_invalidos_se_rechazan(cambio):
    respuesta = cliente_api.post("/predict", json={"clientes": [{**CLIENTE, **cambio}]})
    assert respuesta.status_code == 422


def test_lote_vacio_se_rechaza():
    respuesta = cliente_api.post("/predict", json={"clientes": []})
    assert respuesta.status_code == 422


def test_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_modelo(tmp_path / "no_existe.joblib")
        