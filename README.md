# Actividad 5 — Predicción de tarifas Uber

## Objetivo
Predecir `fare_amount` (USD) a partir de coordenadas, pasajeros, hora y distancia geodésica. Comparar Ridge y Random Forest con GridSearchCV (5 folds) y registrar experimentos en MLflow.

## Fuente y adquisición
Dataset: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
Descargar `uber.csv` desde Kaggle y se coloca en `datos/datos_ini/uber.csv`
**No se incluye CSV limpio hasta ejecutar.** 

## Diccionario de datos
- `fare_amount`: tarifa en USD, objetivo.
- `pickup_datetime`: fecha y hora del inicio (UTC); derivamos hora, día, mes, año.
- `pickup_longitude`, `pickup_latitude`: coordenadas del origen.
- `dropoff_longitude`, `dropoff_latitude`: coordenadas del destino.
- `passenger_count`: cantidad de pasajeros.
- `key`: identificador, descartado; `Unnamed: 0`: índice redundante, descartado.
- `distancia_km`: distancia Haversine entre origen y destino, no distancia real de carretera.

## Instalación y ejecución
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fuentes/train.py --max-rows 30000
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Abrir http://127.0.0.1:5000 y capturar experimentos, métricas, parámetros, modelos y artefactos. Para ejecutar con todos los registros: `python fuentes/train.py --max-rows 0`.

## Reglas y supuestos
Se eliminan duplicados, filas con datos críticos nulos y valores fuera de rangos explícitos: tarifa 2.5–250 USD, 1–6 pasajeros, área NYC aproximada (latitud 40.4–41.1; longitud -74.5 a -73.5), distancia 0.1–80 km. Son supuestos analíticos, no políticas oficiales de Uber. Se convierten tipos, se extraen variables temporales y se calcula distancia Haversine. No hay variables categóricas nominales en las características seleccionadas; día y mes se modelan como enteros. Se aplica escalado StandardScaler **dentro del pipeline Ridge**, evitando fuga de datos en CV; RF no necesita escalado. Imputación mediana dentro de cada pipeline por robustez.

## Evaluación y reproducibilidad
Separación 80/20 con semilla 42. GridSearchCV de 5 folds solo en entrenamiento; conjunto test reservado. Métricas: MAE, RMSE y R², más RMSE medio/desviación en CV. Comparación exploratoria con prueba de Wilcoxon pareada sobre errores absolutos de test (limitaciones por dependencia y tamaño de muestra). MLflow registra hashes SHA-256, hiperparámetros, métricas, auditoría, gráfica y modelo serializado. El directorio `mlruns` se conserva localmente, no se publica por defecto; exportar capturas a `evidencias/`.

## Limitaciones y MLOps
El dataset puede tener sesgo geográfico/temporal; no incluye tráfico, ruta real ni tarifas dinámicas. Las reglas de limpieza pueden excluir trayectos válidos. Próximos pasos: pruebas unitarias, validación de esquema, CI, registro de modelos, monitorización de deriva y reentrenamiento. No interpretar correlación como causalidad.
