# 📊 GUÍA DE INTEGRACIÓN: Dashboard Corregido con ML

## ✅ Problemas Resueltos

### 1. **Coordenadas Inválidas en el Mapa**
**Problema:** Las coordenadas hardcodeadas estaban fuera de Corrientes Capital
**Solución:** Función `extraer_coordenadas_validas()` que:
- Filtra NaN en latitud/longitud
- Valida rango geográfico: ±0.15° del centro (-27.478, -58.825)
- Convierte a numérico y elimina outliers (0,0)
- Agrupa por densidad (bins de 100m x 100m)
- Normaliza intensidad de calor (0-1)

### 2. **No se Integraban Datos del Dataset**
**Problema:** Dashboard ignoraba los 3,781 registros con coordenadas reales
**Solución:** Nueva ruta Flask `/get-coordenadas` que:
- Retorna coordenadas del dataset preprocesado
- Carga dinámicamente en el mapa
- Actualiza al cargar la página

### 3. **Mala Alineación de Vectores ML**
**Problema:** Vectores de entrada no tenían las 176 columnas esperadas
**Solución:** Función `generar_vector_entrada()` mejorada:
- One-Hot Encoding correcto para cada variable
- Reindexación con `X_train.columns`
- Relleno con ceros de dimensiones faltantes

### 4. **Mapa de Calor Estático**
**Problema:** No se actualizaba según predicciones
**Solución:** Método `setOptions()` que ajusta dinámicamente:
- Radio de dispersión: 15 + (promedio/100) × 20
- Blur: 12 + (promedio/100) × 12
- Intensidad máxima: 0.4 + (promedio/100) × 0.6

### 5. **Sin Gestión de Valores Faltantes**
**Problema:** Coordenadas nulas causaban errores
**Solución:** Validación estricta con `notna()`, conversión numérica y filtrado de outliers

---

## 🚀 Pasos de Integración en Colab

### **Paso 1: Después de tu preprocesamiento, cargar el dataset**
```python
import pandas as pd

# Cargar datos preprocesados (después de limpieza y feature engineering)
df = pd.read_csv('tu_archivo_preprocesado.csv')  # O desde tu variable df

# Verificar que tiene las columnas necesarias
print(df[['latitud', 'longitud']].describe())
print(f"Registros con coordenadas válidas: {df[['latitud', 'longitud']].notna().sum()}")
```

### **Paso 2: Copiar el archivo corregido al proyecto**
```python
# En tu notebook, descargar el archivo corregido
!wget -q https://raw.githubusercontent.com/Mar-Lana/Proyecto-Final-2026/main/dashboard_siniestralidad_corregido.py

# O copiar el contenido directamente en una celda
```

### **Paso 3: Generar cache de coordenadas ANTES de iniciar el servidor**
```python
# IMPORTANTE: Ejecutar ANTES de iniciar el dashboard
exec(open('dashboard_siniestralidad_corregido.py').read())

# Generar el cache de coordenadas desde tu dataset preprocesado
coordenadas_cache = extraer_coordenadas_validas(df)
print(f"✅ {len(coordenadas_cache)} puntos de calor generados desde dataset")
```

### **Paso 4: Cargar tus modelos ML entrenados**
```python
import pickle
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Cargar modelos entrenados
rf_model = pickle.load(open('modelo_rf.pkl', 'rb'))      # O tu variable rf_model
xgb_model = pickle.load(open('modelo_xgb.pkl', 'rb'))    # O tu variable xgb_model

# Cargar matriz de entrenamiento para alinear vectores
X_train = pickle.load(open('X_train.pkl', 'rb'))         # O tu variable X_train

print(f"✅ RF model: {rf_model}")
print(f"✅ XGB model: {xgb_model}")
print(f"✅ X_train shape: {X_train.shape}")
```

### **Paso 5: Iniciar el servidor Flask**
```python
# El servidor ya se inició automáticamente al ejecutar el archivo
# Si está en Colab, verás una URL como:
# https://xxxxxx-5000.colab.googleusercontent.com/

print("🌍 Dashboard listo en la URL mostrada arriba")
```

---

## 📝 Estructura del Código Corregido

```
dashboard_siniestralidad_corregido.py
│
├─ 1️⃣ extraer_coordenadas_validas(df_procesado, radio_km=0.15)
│  ├─ Limpieza de NaN
│  ├─ Filtrado geográfico: ±0.15° (≈15 km)
│  ├─ Validación de tipo numérico
│  ├─ Eliminación de outliers (0,0)
│  ├─ Agrupamiento por densidad (bins de 100m)
│  └─ Normalización de intensidad (0-1)
│
├─ 2️⃣ generar_vector_entrada(dia_semana, hora, clima, via, semaforo, colision)
│  ├─ One-Hot Encoding correcto
│  ├─ Cálculo de variables derivadas
│  ├─ Reindexación con X_train.columns
│  └─ Relleno con ceros
│
├─ 3️⃣ Rutas Flask:
│  ├─ GET  / → HTML del dashboard
│  ├─ GET  /get-coordenadas → Retorna coordenadas validadas
│  └─ POST /predict → Ejecuta predicción ML
│
└─ 4️⃣ Frontend HTML/JS:
   ├─ Mapa interactivo con Leaflet
   ├─ Heatmap dinámico (Leaflet.heat)
   ├─ Panel de control con selectores
   ├─ Gráfico de barras (Chart.js)
   └─ Actualización en tiempo real
```

---

## 🔧 Variables Clave de Configuración

```python
# Centro de Corrientes Capital (línea 35)
CENTRO_LAT, CENTRO_LON = -27.478, -58.825

# Radio de filtrado (km)
radio_km = 0.15  # ±15 km aproximadamente

# Tamaño del bin de densidad (grados)
df_limpio['lat_bin'] = (df_limpio['latitud'] // 0.001 * 0.001)  # ≈ 100m × 100m

# Parámetros del heatmap (línea 256-262)
radius: 25          # Radio base de dispersión
blur: 18            # Difuminado
max: 1.0            # Intensidad máxima
gradient: { ... }   # Paleta: azul → cian → verde → amarillo → rojo
```

---

## ⚠️ Errores Comunes y Soluciones

### Error: "X_train not found"
```python
# SOLUCIÓN: Asegúrate de que X_train está en memoria ANTES de ejecutar predict
X_train = your_preprocessed_features  # Matriz de características de entrenamiento
print(X_train.columns)  # Verificar que existen 176 columnas
```

### Error: "No coordenadas válidas"
```python
# SOLUCIÓN: Verificar formato y rango de coordenadas
print(df[['latitud', 'longitud']].describe())
print(df[(df['latitud'].isna()) | (df['longitud'].isna())].shape)

# Limpiar antes de pasar a extraer_coordenadas_validas()
df = df.dropna(subset=['latitud', 'longitud'])
```

### Error: "Modelos no cargados"
```python
# SOLUCIÓN: Cargar antes de iniciar el servidor
global rf_model, xgb_model, X_train
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
X_train = pickle.load(open('X_train.pkl', 'rb'))
```

### Mapa muestra zona vacía
```python
# SOLUCIÓN: Verificar que coordenadas están dentro de rango
print(coordenadas_cache[:5])  # Ver primeros 5 puntos
# Deben estar cerca de: [-27.478, -58.825]
```

---

## 📊 Ejemplo de Salida Esperada

```
================================================================================
✅ DASHBOARD DE SINIESTRALIDAD INICIADO
================================================================================

🌍 URL: https://abc123-5000.colab.googleusercontent.com/

[INFO] Total de registros con coordenadas: 3781
[INFO] Registros dentro del área de Corrientes: 3720
[INFO] Puntos únicos de calor generados: 245

✅ Random Forest model: RandomForestClassifier(...)
✅ XGB model: XGBClassifier(...)
✅ X_train shape: (2664, 176)

================================================================================
```

---

## 🎯 Checklist de Verificación

- [ ] Dataset preprocesado cargado con columnas `latitud` y `longitud`
- [ ] Función `extraer_coordenadas_validas()` retorna >100 puntos
- [ ] Modelos ML (RF y XGB) cargados en memoria
- [ ] Matriz `X_train` tiene 176 columnas
- [ ] Dashboard URL accesible en navegador
- [ ] Mapa muestra puntos rojos/amarillos en Corrientes
- [ ] Predicción ejecuta sin errores
- [ ] Mapa actualiza dinámicamente según nivel de riesgo

---

## 📚 Documentación Adicional

- **Leaflet.heat**: https://www.npmjs.com/package/leaflet.heat
- **Flask**: https://flask.palletsprojects.com/
- **scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/

---

**Autor:** Copilot GitHub  
**Fecha:** Junio 2026  
**Proyecto:** Análisis Predictivo de Siniestralidad Vial - UNNE
