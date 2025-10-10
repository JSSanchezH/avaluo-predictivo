# PROYECTO: Predicción de Precios de Casas en Boston

**Integrante:** Juan Sebastián Sánchez Hincapié

---

## METODOLOGÍA CRISP-DM

---

### 1. ENTENDIMIENTO DEL NEGOCIO

#### 1.1 Descripción del problema

El objetivo principal de este proyecto es **predecir el precio medio de las viviendas en la ciudad de Boston** a partir de diferentes características socioeconómicas y urbanísticas.
Esto permitirá **comprender los factores que más influyen en el valor de una propiedad** y construir un modelo predictivo útil para la toma de decisiones en el sector inmobiliario.

#### 1.2 Diseño de solución

| Tipo de análisis | Tipo de aprendizaje | Posibles métodos                                   | Evaluación      |
| ---------------- | ------------------- | -------------------------------------------------- | --------------- |
| Predictivo       | Supervisado         | KNN, MLP, Regresión Lineal, Árbol de Decisión, SVR | MAE, MAPE, RMSE |

---

### 2. ENTENDIMIENTO DE LOS DATOS

#### 2.1 Variable dependiente

La variable dependiente es aquella que se desea predecir, es desconocida en el futuro.

| Variable | Descripción                                                                       |
| -------- | --------------------------------------------------------------------------------- |
| **MEDV** | Valor medio de las viviendas ocupadas por sus propietarios (en miles de dólares). |

#### 2.2 Variables independientes

Las variables conocidas en el futuro, que permitirán hacer la predicción de la variable dependiente.

| Variable | Descripción                                                                       |
| -------- | --------------------------------------------------------------------------------- |
| CRIM     | Tasa de criminalidad per cápita por ciudad.                                       |
| ZN       | Proporción de terrenos residenciales destinados a lotes de más de 25,000 pies².   |
| INDUS    | Proporción de acres de negocios no minoristas por ciudad.                         |
| CHAS     | Variable ficticia (1 si el tramo limita con el río Charles; 0 en caso contrario). |
| NOX      | Concentración de óxidos nítricos (partes por 10 millones).                        |
| RM       | Número promedio de habitaciones por vivienda.                                     |
| AGE      | Proporción de unidades ocupadas por sus propietarios construidas antes de 1940.   |
| DIS      | Distancia ponderada a cinco centros de empleo en Boston.                          |
| RAD      | Índice de accesibilidad a autopistas radiales.                                    |
| TAX      | Tasa del impuesto a la propiedad por cada 10,000 dólares.                         |
| PTRATIO  | Proporción alumno-profesor por ciudad.                                            |
| B        | 1000(Bk - 0.63)² donde Bk es la proporción de personas negras por ciudad.         |
| LSTAT    | Porcentaje de población de bajo nivel socioeconómico.                             |

#### 2.3 Reglas de calidad

##### Variables numéricas

| Variable | Valor mínimo\* | Valor máximo\* |
| -------- | -------------- | -------------- |
| CRIM     | 0              | 90             |
| ZN       | 0              | 100            |
| INDUS    | 0              | 30             |
| NOX      | 0              | 1              |
| RM       | 3              | 9              |
| AGE      | 0              | 100            |
| DIS      | 0              | 13             |
| RAD      | 1              | 24             |
| TAX      | 100            | 800            |
| PTRATIO  | 10             | 40             |
| BLACK    | 0              | 400            |
| LSTAT    | 0              | 40             |

##### Variables categóricas

| Variable | Valores posibles |
| -------- | ---------------- |
| CHAS     | 0, 1             |

> _Los valores mínimo y máximo son definidos con base en el entendimiento del negocio, no necesariamente los del dataset._

---

### 3. PREPARACIÓN DE DATOS

#### 3.1 Selección de variables desde el conocimiento del negocio

Se seleccionan las variables que más influyen en el precio de las viviendas según estudios previos y correlaciones.

#### 3.2 Descripción estadística

Se realiza un análisis descriptivo con medidas como media, desviación estándar y percentiles para detectar posibles anomalías.

#### 3.3 Limpieza de atípicos

Se identifican y corrigen valores fuera de rango mediante inspección estadística y gráfica (boxplots).

#### 3.4 Limpieza de nulos

No se presentan valores nulos en el dataset original.

#### 3.5 Creación de nuevas variables

Se aplicó **escalado y normalización** de variables numéricas mediante `StandardScaler`.
También se aplicó **One-Hot Encoding** para variables categóricas (ejemplo: CHAS).

#### 3.6 Análisis de relaciones

- **Análisis lineal:** correlación de Pearson para detectar relaciones lineales con `MEDV`.
- **Análisis no lineal:** uso de gráficos de dispersión y modelos de árbol para detectar relaciones no lineales.

#### 3.7 Reducción de dimensiones

Se evaluó la posibilidad de aplicar **PCA**, aunque no fue necesario dado el número reducido de variables.

#### 3.8 Balanceo

No aplica (problema de regresión).

---

### 4. MODELAMIENTO Y EVALUACIÓN

#### 4.1 Configuración y selección de métodos de Machine Learning

##### 4.1.1 Modelos clásicos seleccionados

- Regresión Lineal
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- MLP Regressor

##### 4.1.2 Modelos de Ensamble

- **Votación (Voting Regressor)**
- **Bagging (Random Forest)**
- **Boosting (Gradient Boosting)**

#### 4.2 Ajuste de hiperparámetros

Se usó el **70% de los datos para entrenamiento** y **30% para prueba**.
Se aplicó **Cross Validation (k=5)** para optimizar hiperparámetros.

##### 4.2.1 Justificación de la métrica

Se seleccionó **MAE** y **MAPE** por su interpretabilidad en unidades monetarias y porcentuales.

##### 4.2.2 Ajuste de modelos clásicos

GridSearchCV se empleó para ajustar hiperparámetros como:

- `n_neighbors` (KNN)
- `max_depth` (Árboles)
- `hidden_layer_sizes` y `alpha` (MLP)

##### 4.2.3 Ajuste de modelos de ensamble

Se optimizaron parámetros como número de árboles (`n_estimators`) y tasa de aprendizaje (`learning_rate`).

#### 4.3 Medida de calidad del modelo

##### 4.3.1 Evaluación con set de pruebas

El modelo final se evaluó con métricas MAE, MAPE y RMSE.

##### 4.3.2 Selección del mejor modelo

El **Gradient Boosting Regressor** obtuvo el mejor desempeño general.

---

### 5. DESPLIEGUE

#### 5.1 Construcción del conjunto de datos de entrada

Se preparó un conjunto de datos nuevo con características simuladas o reales.

#### 5.2 Preparación del conjunto de datos

Se aplicaron los mismos pasos de escalado y codificación usados en el entrenamiento.

#### 5.3 Predicción con el modelo

El modelo predice el valor medio de una vivienda en miles de dólares.
Por ejemplo, una vivienda con **RM = 6.5**, **LSTAT = 10**, **DIS = 5** puede tener un valor estimado de **$28,000 USD**.

#### 5.4 Despliegue en aplicación web

El modelo fue implementado usando **Streamlit** y desplegado en **Streamlit Cloud**.
La aplicación permite ingresar datos manualmente o cargar archivos `.csv` para obtener predicciones en tiempo real.

[Haz clic aquí para abrir la aplicación](https://multipage-a96sukjk59keo5m9i7tnbg.streamlit.app/precios_casas_boston)

---

### 💻 Tecnologías utilizadas

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

### 👨‍💻 Autor

**Juan Sebastián Sánchez Hincapié**
[LinkedIn](https://www.linkedin.com/in/jssanchezh/)

---
