# Importar librerías
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la aplicación
st.set_page_config(page_title="Predicción de Propiedades en Boston", layout="wide", page_icon="🏡")
st.title("🏡 Análisis y Predicción de Valores de Propiedades en Boston")
st.write("Explora los factores que influyen en el valor de las propiedades en Boston y realiza predicciones con un modelo de regresión lineal.")

# Cargar los datos
df = pd.read_csv('boston.csv').drop('Unnamed: 0', axis=1)

# Pestañas para la navegación de la aplicación
tabs = st.tabs(["📊 Exploración de Datos", "📈 Modelo y Coeficientes", "📉 Evaluación del Modelo", "🔮 Predicción Interactiva"])

with tabs[0]:  # Exploración de datos
    st.header("Exploración de Datos")
    st.write("Visualiza los primeros registros y las estadísticas descriptivas del conjunto de datos de Boston.")
    st.write(df.head())
    
    with st.expander("Ver Estadísticas Descriptivas"):
        st.write(df.describe())
    
    # Gráfico de correlación
    st.subheader("Correlación de Características con 'medv'")
    st.write("Las características están ordenadas por su correlación con el valor objetivo ('medv'):")
    
    def fetch_features(dataframe, vector_objetivo='medv'):
        columns = dataframe.columns
        attr_name = []
        pearson_r = []  
        abs_pearson_r = []
        for col in columns:
            if col != vector_objetivo:
                attr_name.append(col)
                pearson_r.append(dataframe[col].corr(dataframe[vector_objetivo]))
                abs_pearson_r.append(abs(dataframe[col].corr(dataframe[vector_objetivo])))
        features = pd.DataFrame({'attribute': attr_name, 'corr': pearson_r, 'abs_corr': abs_pearson_r})
        return features.set_index('attribute').sort_values(by=['abs_corr'], ascending=False)

    correlation_features = fetch_features(df)
    st.write(correlation_features)
    
    # Gráfico de barras de correlación
    fig, ax = plt.subplots(figsize=(8, 4))
    correlation_features['corr'].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Correlación de Características con 'medv'")
    ax.set_ylabel("Coeficiente de Correlación")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tabs[1]:  # Modelo y Coeficientes
    st.header("Modelo de Regresión Lineal y Coeficientes")
    y = df['medv']
    X = df[['lstat', 'rm', 'ptratio', 'indus', 'tax', 'nox']]
    
    # División de datos
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Crear y entrenar el modelo
    modelo = linear_model.LinearRegression()
    modelo.fit(x_train, y_train)
    
    # Coeficientes del modelo
    coef_df = pd.DataFrame({"Característica": X.columns, "Coeficiente": modelo.coef_})
    st.write(coef_df)
    
    # Gráfico de coeficientes
    fig, ax = plt.subplots(figsize=(8, 4))
    coef_df.set_index('Característica').plot(kind='bar', ax=ax, legend=False, color='lightcoral')
    ax.set_title("Coeficientes del Modelo")
    ax.set_ylabel("Valor del Coeficiente")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tabs[2]:  # Evaluación del Modelo
    st.header("Evaluación del Modelo")
    
    # Predicciones y cálculo de métricas
    modelo_yhat = modelo.predict(x_test)
    mse = mean_squared_error(y_test, modelo_yhat)
    r2 = r2_score(y_test, modelo_yhat)
    
    # Métricas en columnas con estilo
    col1, col2 = st.columns(2)
    col1.metric("📉 Error Cuadrático Medio (MSE)", f"{mse:.2f}")
    col2.metric("📊 Coeficiente de Determinación (R²)", f"{r2:.2f}")
    
    # Gráfico de Predicciones vs Valores Reales
    st.subheader("Comparación entre Predicciones y Valores Reales")
    fig, ax = plt.subplots()
    ax.scatter(y_test, modelo_yhat, alpha=0.7, color='mediumseagreen')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Predicciones")
    ax.set_title("Predicciones vs Valores Reales")
    st.pyplot(fig)

with tabs[3]:  # Predicción Interactiva
    st.header("Simulación de Valor para Nuevos Vecindarios")
    st.write("Ingrese las características del vecindario para predecir su valor.")
    
    # Predicción para el peor vecindario
    worst_values = [37.9, 3.5, 12.6, 27.7, 187, 0.87]
    worst_neighbor = modelo.predict(np.array(worst_values).reshape(1, -1))
    st.info(f"Predicción para el peor vecindario: **${worst_neighbor[0]:,.2f}**")
    
    # Formulario de predicción interactiva
    with st.form(key="vecindario_form"):
        values = []
        for i, col in enumerate(X.columns):
            values.append(st.number_input(f"{col}", value=float(worst_values[i])))
        submit_button = st.form_submit_button(label="Predecir Valor")
    
        if submit_button:
            input_data = np.array(values).reshape(1, -1)
            prediccion = modelo.predict(input_data)
            st.success(f"Valor predicho para el vecindario ingresado: **${prediccion[0]:,.2f}**")
            
# Nota informativa
st.info("Esta aplicación proporciona un análisis basado en un modelo de regresión lineal y se enfoca en los factores de correlación más altos para predecir el valor de propiedades en Boston.")
