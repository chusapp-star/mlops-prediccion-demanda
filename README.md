# Centro Inteligente de Predicción de Demanda - Laboratorio 4 MLOps

Proyecto académico del curso **Analítica Predictiva y Aprendizaje Automático**. La solución transforma un ejercicio de modelado en un flujo MLOps reproducible y orientado a un usuario empresarial.

**Estudiante:** Jesús Prado Portuguez  
**Laboratorio:** 4 - Módulo 4  
**Caso:** Predicción de demanda empresarial

## Objetivo

Estimar `Demanda_Dia_Siguiente` para apoyar decisiones de inventario, abastecimiento, promociones, planificación y asignación de recursos.

## Resultado principal

Se compararon cuatro modelos bajo el mismo pipeline de preprocesamiento:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 5.5702 | 7.2443 | 0.8124 |
| Random Forest | 5.6844 | 7.2919 | 0.8100 |
| Regresión Lineal | 6.9477 | 8.8125 | 0.7224 |
| Baseline | 13.3046 | 16.7294 | -0.0003 |

Se seleccionó **Gradient Boosting** porque obtuvo el menor MAE y RMSE, además del mayor R² entre los modelos evaluados.

## Calidad de datos

- Registros originales: 1,818
- Registros duplicados: 18
- Registros luego de eliminar duplicados: 1,800
- Valores faltantes: 68
  - `Temperatura_C`: 36
  - `Visitas_Web`: 32
- División entrenamiento/prueba: 1,350 / 450 registros

## Variables excluidas para evitar data leakage

`ID_Registro`, `Fecha`, `Demanda_Dia_Siguiente`, `Demanda_Predicha_Previa`, `Error_Absoluto_Previa`, `Version_Modelo`, `Latencia_ms`, `Drift_Score` y `Alerta_Drift`.

Las variables de desempeño de modelos previos se reservan para monitoreo y no se utilizan como predictores del nuevo modelo.

## Estructura del proyecto

- `app.py`: aplicación Streamlit.
- `modelo_demanda_produccion.joblib`: pipeline completo serializado.
- `metadata_modelo.json`: versión, métricas y datos de entrenamiento.
- `requirements.txt`: dependencias necesarias.
- `datos_nuevos_ejemplo.csv`: ejemplo listo para probar la aplicación.
- `Comparacion_Modelos_Demanda.csv`: métricas de los modelos evaluados.
- `Importancia_Variables.csv`: importancia predictiva por permutación.
- `Monitoreo_Modelo_Demanda.csv`: evidencia de predicciones y error en prueba.
- `Analisis_Errores.csv`: casos ordenados por error absoluto.
- `Laboratorio_4_MLOps_Demanda_Ejecutado.ipynb`: notebook con evidencia reproducible.
- `Reporte_Tecnico_Laboratorio_4_MLOps.docx/pdf`: informe final.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso de la aplicación

1. Abra la sección **Nueva predicción**.
2. Descargue el archivo de ejemplo o cargue un CSV con la estructura del laboratorio.
3. Presione **Generar predicciones**.
4. Revise el resultado y descargue el CSV generado.

La app acepta tanto registros ya preparados como la estructura cruda del laboratorio: transforma `Fecha`, convierte `Descuento_Porc` y elimina columnas que no deben entrar al modelo.

## Monitoreo y reentrenamiento

El MAE de referencia es 5.57 unidades. Como criterio de advertencia se propone revisar el modelo cuando el MAE operativo supere aproximadamente **6.96 unidades** (25% por encima de la referencia). No se recomienda reentrenar por un solo valor aislado: se debe confirmar deterioro sostenido, data drift persistente, aparición de nuevas categorías/canales o cambios importantes en el comportamiento de la demanda.

## Uso responsable

La predicción es un apoyo para la decisión empresarial y no sustituye el criterio profesional. No deben publicarse credenciales, tokens, contraseñas, datos personales ni información confidencial. La importancia predictiva de una variable tampoco debe interpretarse automáticamente como causalidad.
