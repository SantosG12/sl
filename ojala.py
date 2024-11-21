import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la aplicación
st.title("Predicción de Problemas Cardiovasculares")
st.write("Ingrese los datos del paciente para predecir la probabilidad de problemas cardiovasculares.")

# Cargar el modelo y el preprocesador
try:
    modelo_cargado = joblib.load('modelo_ganador_logreg(1).pkl')
    preprocessor_cargado = joblib.load('preprocessor.pkl')
    st.success("Modelo y preprocesador cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo o el preprocesador: {e}")
    st.stop()

# Interfaz para ingresar datos manualmente
st.sidebar.header("Ingrese los datos del paciente")
genero = st.sidebar.selectbox("Género", ["Male", "Female"])
colesterol = st.sidebar.selectbox("Colesterol (1=Normal, 2=Elevado, 3=Muy Elevado)", ['1', '2', '3'])
glucosa = st.sidebar.selectbox("Glucosa (1=Normal, 2=Elevada, 3=Muy Elevada)", ['1', '2', '3'])
fuma = st.sidebar.radio("¿Fuma?", ['0', '1'])
toma_alcohol = st.sidebar.radio("¿Toma Alcohol?", ['0', '1'])
actividad_fisica = st.sidebar.radio("¿Realiza Actividad Física?", ['0', '1'])
edad = st.sidebar.number_input("Edad (años)", min_value=1, max_value=120, value=30, step=1)
altura = st.sidebar.number_input("Altura (cm)", min_value=50, max_value=250, value=160, step=1)
peso = st.sidebar.number_input("Peso (kg)", min_value=10, max_value=300, value=70, step=1)
presion_sistolica = st.sidebar.number_input("Presión Sistólica (mmHg)", min_value=50, max_value=300, value=120, step=1)
presion_diastolica = st.sidebar.number_input("Presión Diastólica (mmHg)", min_value=30, max_value=200, value=80, step=1)

# Calcular BMI
bmi = round(peso / ((altura / 100) ** 2), 2)

# Crear DataFrame con los datos ingresados
nuevos_datos = pd.DataFrame({
    'Genero': [genero],
    'Colesterol': [colesterol],
    'Glucosa': [glucosa],
    'Fuma': [fuma],
    'Toma_alchol': [toma_alcohol],
    'Actividad_fisica': [actividad_fisica],
    'Edad': [edad],
    'Altura': [altura],
    'Peso': [peso],
    'Presion_arterial_sistolica': [presion_sistolica],
    'Presion_arterial_diastolica': [presion_diastolica],
    'Bmi': [bmi]
})

# Transformar los datos con el preprocesador
try:
    nuevos_datos_transformados = preprocessor_cargado.transform(nuevos_datos)

    # Hacer la predicción
    prediccion = modelo_cargado.predict(nuevos_datos_transformados)
    probabilidad = modelo_cargado.predict_proba(nuevos_datos_transformados)[:, 1]

    # Mostrar los resultados
    st.subheader("Resultado de la Predicción")
    if prediccion[0] == 1:
        st.error("El modelo predice que el paciente tiene riesgo de problemas cardiovasculares.")
    else:
        st.success("El modelo predice que el paciente NO tiene riesgo de problemas cardiovasculares.")
    st.write(f"Probabilidad estimada de riesgo: {probabilidad[0]:.2%}")
except Exception as e:
    st.error(f"Error al realizar la predicción: {e}")
