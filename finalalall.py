import joblib
import streamlit as st

# Cargar modelo entrenado con joblib
modelo_path = 'modelo_ganador_logreg.pkl'

# Verificar si el archivo existe
try:
    modelo_cargado = joblib.load(modelo_path)
    st.success("Modelo cargado exitosamente.")
except FileNotFoundError:
    st.error(f"No se encontró el archivo {modelo_path}. Por favor, verifica que esté en el repositorio.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Interfaz de la aplicación
st.title("Predicción de Problemas Cardiovasculares")
st.write("Ingrese los datos del paciente para predecir la probabilidad de problemas cardiovasculares.")

# Crear la interfaz para ingresar datos
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

# Crear un DataFrame con los datos ingresados
import pandas as pd

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

# Realizar la predicción
try:
    prediccion = modelo_cargado.predict(nuevos_datos)
    probabilidad = modelo_cargado.predict_proba(nuevos_datos)[:, 1]

    st.subheader("Resultado de la Predicción")
    if prediccion[0] == 1:
        st.error("El modelo predice que el paciente tiene riesgo de problemas cardiovasculares.")
    else:
        st.success("El modelo predice que el paciente NO tiene riesgo de problemas cardiovasculares.")
    st.write(f"Probabilidad estimada de riesgo: {probabilidad[0]:.2%}")
except Exception as e:
    st.error(f"Error al realizar la predicción: {e}")
