# Documentación Técnica: AI in HealthCare - Chest Disease Detection
**Autor:** Roberto Alejandro Chagra Martínez  
**Tecnologías:** TensorFlow / Keras, OpenCV, Matplotlib, NumPy  
**Tipo:** Clasificación Radiológica Pulmonar Multiclase  

---

## 1. Descripción del Proyecto
Solución de diagnóstico radiológico asistido por IA para clasificar enfermedades del tórax (Neumonía, Tuberculosis, COVID-19 y controles Sanos) a partir de radiografías digitales (X-Rays).

---

## 2. Flujo de Datos y Componentes
* `procesar_dataset.py`: Normalización de niveles de gris, ecualización de histograma (CLAHE) y redimensionamiento a escala de modelo.
* `reorganizar_dataset.py`: Partición balanceada y estratificada en conjuntos de Train, Validation y Test.
* `train_model.py`: Ajuste fino de arquitectura convolucional profunda con regularización por abandono (Dropout).
* `evaluate_model.py`: Reporte de métricas diagnósticas y análisis de falsos negativos / falsos positivos.
* `app.py`: Interfaz de usuario para carga e inferencia de radiografías.
