# AutoML h20
# Combinar los principales algoritmos de machine learning y aprendizaje estadístico con el Big Data. 

# Importamos las librerías
library(h2o)
library(pROC)
library(caret)

# Arrancamos el cluster h20 (-1 indica que se empleen todos los cores disponibles)
h2o.init(nthreads = -1) 


# Arreglamos los datos train y eliminamos las columnas que nos necesitamos
train_data <- readxl::read_xlsx("datosPractica/Datos_Train80.xlsx")
dim(train_data); names(train_data)
train_data$...1 <- NULL
train_data$is_train <- NULL
train_data$sid <- NULL
dim(train_data); names(train_data)

# Carga de datos en el cluster H2O 
train.h2o <- as.h2o(train_data)

# Convertimos la variable binaryTarget a factor (necesario)
train.h2o['binaryTarget'] <- as.factor(train.h2o['binaryTarget'])


# Arreglamos los datos validation y eliminamos las columnas que nos necesitamos
val_data <- readxl::read_xlsx("datosPractica/Datos_Val20.xlsx")
dim(val_data); names(val_data)
val_data$...1 <- NULL
val_data$is_train <- NULL
val_data$sid <- NULL
y_true <- as.vector(val_data$binaryTarget)
val_data$binaryTarget <- NULL
dim(val_data); names(val_data)

# Carga de datos en el cluster H2O 
val.h2o <- as.h2o(val_data)


# Arreglamos los datos de evaluación y eliminamos las columnas que nos necesitamos
eval_data <- readxl::read_xlsx("datosPractica/Datos_Eval.xlsx")
dim(eval_data); names(eval_data)
eval_data$...1 <- NULL
eval_data$is_train <- NULL
eval_data$sid <- NULL
dim(eval_data); names(eval_data)

# Carga de datos en el cluster H2O 
eval.h2o <- as.h2o(eval_data)


# Separar los conjuntos de datos train de la etiqueta o variable target
inputs <- train_data[, ! names(train_data) %in% c("binaryTarget"), drop = F]
predictors <- names(inputs)
response <- 'binaryTarget'


# Ajuste del modelo y validación mediente 5-CV para estimar su error.
aml <- h2o.automl(x = predictors,
                  y = response, 
                  training_frame = train.h2o,
                  nfolds = 5,
                  max_runtime_secs_per_model = 40,
                  #balance_classes = TRUE,
                  stopping_metric = "misclassification",
                  stopping_rounds = 100,
                  max_models = 25,
                  seed = 3)
aml

# Visualizamos todos los modelos
aml.output <- aml@leaderboard
print(aml.output, n = nrow(aml.output))

# Visualizamos el modelo que ha ganado
aml@leader


# Calulamos el accuracy
pred <- as.data.frame(h2o.predict(aml, val.h2o))
y_pred <- as.vector(pred$predict) 
auc(y_true, as.numeric(y_pred))
confusionMatrix(as.factor(as.numeric(y_pred)), as.factor(y_true))


# Calulamos las predicciones
pred <- as.data.frame(h2o.predict(aml, eval.h2o)); pred
table(factor(pred$predict))
