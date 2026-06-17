"""
DASHBOARD CORREGIDO: Predicción de Siniestralidad Vial - Corrientes
Análisis Predictivo + Mapa de Calor + Modelos ML (Random Forest, XGBoost)

Problemas resueltos:
1. ✅ Validación estricta de coordenadas geográficas (dentro de Corrientes Capital)
2. ✅ Integración correcta de datos preprocesados (latitud/longitud del dataset)
3. ✅ Alineación de vectores de entrada con los modelos ML entrenados
4. ✅ Mapa de calor dinámico basado en predicciones en tiempo real
5. ✅ Gestión robusta de valores faltantes en coordenadas
"""

import os
import threading
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template_string
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==============================================================================
# 1️⃣ EXTRACCIÓN Y VALIDACIÓN DE COORDENADAS DESDE DATASET PREPROCESADO
# ==============================================================================

def extraer_coordenadas_validas(df_procesado, radio_km=0.15):
    """
    Extrae coordenadas válidas del dataset preprocesado.
    
    Parámetros:
    -----------
    df_procesado : DataFrame
        Dataset después de limpieza y preprocesamiento
    radio_km : float
        Radio en km alrededor del centro de Corrientes para filtrado
        
    Retorna:
    --------
    list : Lista de tuplas (lat, lon, intensidad) válidas
    """
    
    # Centro aproximado de Corrientes Capital: -27.478, -58.825
    CENTRO_LAT, CENTRO_LON = -27.478, -58.825
    
    # Filtrar coordenadas válidas
    coordenadas_validas = []
    
    # 1. Limpieza básica de NaN
    df_limpio = df_procesado[
        (df_procesado['latitud'].notna()) & 
        (df_procesado['longitud'].notna())
    ].copy()
    
    print(f"[INFO] Total de registros con coordenadas: {len(df_limpio)}")
    
    # 2. Validar rango geográfico (±0.15 grados ≈ ±15 km)
    df_limpio = df_limpio[
        (df_limpio['latitud'] >= CENTRO_LAT - radio_km) &
        (df_limpio['latitud'] <= CENTRO_LAT + radio_km) &
        (df_limpio['longitud'] >= CENTRO_LON - radio_km) &
        (df_limpio['longitud'] <= CENTRO_LON + radio_km)
    ]
    
    print(f"[INFO] Registros dentro del área de Corrientes: {len(df_limpio)}")
    
    # 3. Convertir a tipo numérico (por si hay strings)
    df_limpio['latitud'] = pd.to_numeric(df_limpio['latitud'], errors='coerce')
    df_limpio['longitud'] = pd.to_numeric(df_limpio['longitud'], errors='coerce')
    
    # 4. Eliminar outliers extremos (coordenadas mal cargadas)
    df_limpio = df_limpio[
        (df_limpio['latitud'] != 0) & 
        (df_limpio['longitud'] != 0)
    ]
    
    # 5. Agrupar por ubicación y calcular densidad de calor
    # Crear bins de 0.001 grados (aproximadamente 100m x 100m)
    df_limpio['lat_bin'] = (df_limpio['latitud'] // 0.001 * 0.001).round(3)
    df_limpio['lon_bin'] = (df_limpio['longitud'] // 0.001 * 0.001).round(3)
    
    densidad = df_limpio.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
    densidad_max = densidad['count'].max()
    
    # Normalizar intensidad (0 a 1)
    densidad['intensidad'] = densidad['count'] / densidad_max
    
    # Crear lista de coordenadas con intensidad
    for _, row in densidad.iterrows():
        coordenadas_validas.append([
            float(row['lat_bin']),
            float(row['lon_bin']),
            float(row['intensidad'])
        ])
    
    print(f"[INFO] Puntos únicos de calor generados: {len(coordenadas_validas)}")
    return coordenadas_validas


# ==============================================================================
# 2️⃣ GENERADOR DE VECTORES DE ENTRADA ALINEADO CON MODELOS ML
# ==============================================================================

def generar_vector_entrada(dia_semana, hora, clima, via, semaforo, colision):
    """
    Crea un vector de entrada compatible con los modelos ML.
    Utiliza One-Hot Encoding para variables categóricas.
    """
    global X_train
    
    datos_base = {
        'anio': 2026,
        'hora': int(hora),
        'mes': 6,
        'vehiculos': 2,
        'peatones': 1 if colision == 'PEATON' else 0
    }
    
    # Variables derivadas
    h = int(hora)
    datos_base['hora_pico'] = 1 if (7 <= h <= 9 or 12 <= h <= 13 or 17 <= h <= 19) else 0
    
    # Franja horaria
    franja = 'Noche'
    if 0 <= h < 6:
        franja = 'Madrugada'
    elif 6 <= h < 12:
        franja = 'Manana'
    elif 12 <= h < 15:
        franja = 'Siesta'
    elif 15 <= h < 20:
        franja = 'Tarde'
    
    # One-Hot Encoding para variables categóricas
    datos_base[f'franja_horaria_{franja}'] = 1
    datos_base[f'dia_semana_{dia_semana}'] = 1
    datos_base[f'estado_fisico_ambiental_{clima}'] = 1
    datos_base[f'tipo_via_macro_{via}'] = 1
    datos_base[f'semaforo_{"EXISTE" if semaforo else "S/D"}'] = 1
    
    if colision != 'OTRO':
        datos_base[f'tipo_siniestro_multiple_{colision}'] = 1
        datos_base[f'tipo_colision_{colision}'] = 1
    
    df_temp = pd.DataFrame([datos_base])
    
    # Alinear con estructura de X_train
    if 'X_train' in globals():
        columnas_esperadas = X_train.columns
        df_pred = df_temp.reindex(columns=columnas_esperadas, fill_value=0)
    else:
        raise ValueError("[ERROR] X_train no cargado. Ejecuta primero el preprocesamiento.")
    
    return df_pred


# ==============================================================================
# 3️⃣ SERVIDOR FLASK CON DASHBOARD MEJORADO
# ==============================================================================

app = Flask("SiniestralidadCorrientes")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Siniestralidad Vial - Corrientes</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        .leaflet-container { font-family: inherit; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
        .spinner { 
            border: 3px solid #334155; 
            border-top: 3px solid #6366f1; 
            border-radius: 50%; 
            width: 20px; 
            height: 20px; 
            animation: spin 1s linear infinite; 
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">

    <!-- Header -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4">
        <div class="flex items-center gap-3">
            <div class="p-2.5 bg-indigo-600/20 text-indigo-500 rounded-lg border border-indigo-500/30">
                <i class="fa-solid fa-fire-flame-curved text-2xl"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold text-white">Observatorio Vial Inteligente - Corrientes</h1>
                <p class="text-xs text-slate-400">Análisis Predictivo basado en Datos Reales (2018-2025)</p>
            </div>
        </div>
    </header>

    <main class="flex-1 p-6 grid grid-cols-1 xl:grid-cols-4 gap-6">
        <!-- Panel de Control -->
        <div class="xl:col-span-1 flex flex-col gap-6">
            <div class="bg-slate-900 rounded-xl border border-slate-800 p-5">
                <h2 class="text-sm font-semibold text-white mb-4">Simulador de Entorno</h2>
                
                <div class="space-y-3 text-xs">
                    <div>
                        <label class="block text-slate-400 mb-1 font-medium">Día de la Semana</label>
                        <select id="sim-dia" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                            <option value="Lunes">Lunes</option>
                            <option value="Viernes" selected>Viernes</option>
                            <option value="Sábado">Sábado</option>
                            <option value="Domingo">Domingo</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-slate-400 mb-1 font-medium">Hora</label>
                        <select id="sim-hora" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                            <option value="4">04:00 (Madrugada)</option>
                            <option value="8">08:00 (Mañana)</option>
                            <option value="13" selected>13:00 (Siesta)</option>
                            <option value="19">19:00 (Tarde)</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-slate-400 mb-1 font-medium">Clima</label>
                        <select id="sim-clima" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                            <option value="DESPEJADA, SECA" selected>Despejado</option>
                            <option value="LLUVIOSO, HUMEDA">Lluvia</option>
                            <option value="NUBLADO">Nublado</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-slate-400 mb-1 font-medium">Tipo de Vía</label>
                        <select id="sim-via" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                            <option value="AVENIDA" selected>Avenida</option>
                            <option value="CALLE">Calle</option>
                            <option value="RUTA">Ruta</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-slate-400 mb-1 font-medium">Tipo de Colisión</label>
                        <select id="sim-colision" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
                            <option value="LATERAL" selected>Lateral</option>
                            <option value="ANGULO">Ángulo</option>
                            <option value="PEATON">Atropello</option>
                            <option value="OTRO">Otro</option>
                        </select>
                    </div>

                    <label class="flex items-center gap-2 cursor-pointer bg-slate-800 p-2.5 rounded-lg border border-slate-700">
                        <input type="checkbox" id="sim-semaforo" class="rounded bg-slate-700">
                        <span>¿Semáforo?</span>
                    </label>
                </div>

                <button onclick="calcularPrediccion()" class="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-lg text-xs transition flex items-center justify-center gap-2">
                    <i class="fa-solid fa-brain"></i> Ejecutar Predicción
                </button>

                <!-- Panel de Resultados -->
                <div id="panel-resultado" class="mt-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <p class="text-[10px] text-slate-400 uppercase mb-2">Predicciones en Tiempo Real</p>
                    <div class="grid grid-cols-2 gap-2">
                        <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
                            <p class="text-[10px] text-slate-400">Random Forest</p>
                            <p id="res-rf" class="text-xl font-bold text-blue-400">-- %</p>
                        </div>
                        <div class="bg-slate-900 p-2 rounded-lg border border-slate-800">
                            <p class="text-[10px] text-slate-400">XGBoost</p>
                            <p id="res-xgb" class="text-xl font-bold text-orange-400">-- %</p>
                        </div>
                    </div>
                    <div id="resultado-etiqueta" class="text-[10px] uppercase font-bold px-2 py-1.5 rounded mt-2 bg-slate-800 text-slate-400">
                        A la espera
                    </div>
                </div>
            </div>
        </div>

        <!-- Mapa de Calor -->
        <div class="xl:col-span-2">
            <div class="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden h-[600px]">
                <div id="map" class="w-full h-full"></div>
            </div>
        </div>

        <!-- Gráficos -->
        <div class="xl:col-span-1">
            <div class="bg-slate-900 p-5 rounded-xl border border-slate-800 h-[600px] flex flex-col">
                <p class="text-xs font-semibold text-white mb-3">Comparativa de Riesgo</p>
                <div class="flex-1">
                    <canvas id="chart-probabilidad"></canvas>
                </div>
                <div class="text-xs text-slate-400 mt-3 text-center">
                    <p class="text-[10px]">Mapa generado con datos reales del dataset 2018-2025</p>
                </div>
            </div>
        </div>
    </main>

    <footer class="bg-slate-900 border-t border-slate-800 px-6 py-3 text-center text-xs text-slate-500">
        © 2026 UNNE • Análisis Predictivo de Siniestralidad Vial
    </footer>

    <script>
        let map;
        let chart;
        let heatLayer = null;
        let coordenadasHistoricas = [];

        window.onload = async function() {
            await cargarCoordenadas();
            initMap();
            initChart();
            calcularPrediccion();
        };

        async function cargarCoordenadas() {
            try {
                const response = await fetch('/get-coordenadas');
                const data = await response.json();
                coordenadasHistoricas = data.coordenadas;
                console.log(`[OK] ${coordenadasHistoricas.length} puntos de calor cargados desde dataset`);
            } catch (err) {
                console.error("Error cargando coordenadas:", err);
            }
        }

        function initMap() {
            map = L.map('map', { zoomControl: true }).setView([-27.478, -58.825], 13);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 20 }).addTo(map);

            // Crear capa de calor inicial
            if (coordenadasHistoricas.length > 0) {
                heatLayer = L.heatLayer(coordenadasHistoricas, {
                    radius: 25,
                    blur: 18,
                    max: 1.0,
                    gradient: {
                        0.2: 'blue',
                        0.4: 'cyan',
                        0.6: 'lime',
                        0.8: 'yellow',
                        1.0: 'red'
                    }
                }).addTo(map);
            }
        }

        function initChart() {
            const ctx = document.getElementById('chart-probabilidad').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Random Forest', 'XGBoost'],
                    datasets: [{
                        data: [0, 0],
                        backgroundColor: ['#3b82f6', '#f97316'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { min: 0, max: 100, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        }

        async function calcularPrediccion() {
            const data = {
                dia_semana: document.getElementById('sim-dia').value,
                hora: document.getElementById('sim-hora').value,
                clima: document.getElementById('sim-clima').value,
                via: document.getElementById('sim-via').value,
                colision: document.getElementById('sim-colision').value,
                semaforo: document.getElementById('sim-semaforo').checked
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                if (result.success) {
                    const rfVal = result.rf_proba * 100;
                    const xgbVal = result.xgb_proba * 100;
                    const promedio = (rfVal + xgbVal) / 2;

                    document.getElementById('res-rf').innerText = rfVal.toFixed(1) + '%';
                    document.getElementById('res-xgb').innerText = xgbVal.toFixed(1) + '%';

                    chart.data.datasets[0].data = [rfVal, xgbVal];
                    chart.update();

                    // Actualizar etiqueta
                    const etiq = document.getElementById('resultado-etiqueta');
                    if (promedio < 45) {
                        etiq.innerText = "✓ Riesgo Bajo";
                        etiq.className = "text-[10px] uppercase font-bold px-2 py-1.5 rounded bg-emerald-500/20 text-emerald-300";
                    } else if (promedio < 70) {
                        etiq.innerText = "⚠ Riesgo Moderado";
                        etiq.className = "text-[10px] uppercase font-bold px-2 py-1.5 rounded bg-amber-500/20 text-amber-300";
                    } else {
                        etiq.innerText = "🔴 Riesgo Crítico";
                        etiq.className = "text-[10px] uppercase font-bold px-2 py-1.5 rounded bg-red-500/20 text-red-300";
                    }

                    // Actualizar mapa dinámicamente
                    if (heatLayer && coordenadasHistoricas.length > 0) {
                        const nuevoRadio = 15 + (promedio / 100) * 20;
                        const nuevoBlur = 12 + (promedio / 100) * 12;
                        const nuevaIntensidad = Math.min(1.0, 0.4 + (promedio / 100) * 0.6);

                        heatLayer.setOptions({
                            radius: nuevoRadio,
                            blur: nuevoBlur,
                            max: nuevaIntensidad
                        });
                    }
                }
            } catch (err) {
                console.error("Error en predicción:", err);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get-coordenadas')
def get_coordenadas():
    """Retorna las coordenadas validadas del dataset"""
    global coordenadas_cache
    if 'coordenadas_cache' in globals():
        return jsonify({'coordenadas': coordenadas_cache})
    return jsonify({'coordenadas': []})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X_nuevo = generar_vector_entrada(
            dia_semana=data.get('dia_semana'),
            hora=data.get('hora'),
            clima=data.get('clima'),
            via=data.get('via'),
            semaforo=data.get('semaforo'),
            colision=data.get('colision')
        )

        # Usar modelos reales del contexto global
        global rf_model, xgb_model
        prob_rf = float(rf_model.predict_proba(X_nuevo)[0][1])
        prob_xgb = float(xgb_model.predict_proba(X_nuevo)[0][1])

        return jsonify({
            'success': True,
            'rf_proba': prob_rf,
            'xgb_proba': prob_xgb
        })
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# ==============================================================================
# 4️⃣ INICIALIZACIÓN
# ==============================================================================

def iniciar_servidor():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

servidor_thread = threading.Thread(target=iniciar_servidor, daemon=True)
servidor_thread.start()

print("="*80)
print("✅ DASHBOARD DE SINIESTRALIDAD INICIADO")
print("="*80)

if IN_COLAB:
    url_publica = output.eval_js("google.colab.kernel.proxyPort(5000)")
    print(f"\n🌍 URL: {url_publica}\n")
else:
    print("\n🌍 URL: http://localhost:5000\n")

print("="*80)
