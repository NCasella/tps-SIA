# tp4-SIA_Unsupervised-Learning

Todas las dependencias necesarias estan dentro del archivo ```requirements.txt```, que se pueden instalar con:

``pip install -r requirements.txt``

## Red de Kohonen
Para la red de Kohonen, ejecutar:

``python3 ej1.1kohonen.py config/ej1.1config.json``

Dentro del archivo ```ej1.1config.json``` de configuracion, se encuentran los siguentes parametros:

- grid_size: integer
- data_source: path
- iterations: integer
- learning_rate: float [0,1]
- similariry_metric: ["euclidean_distance","exponential_distance"]
- random_weights: [true, false]
- radius: integer
- constat_radius: [true, false]

## Red de Oja
Para la red de oja, ejecutar:

``python3 ej1.2oja.py config/ej1.2config.json``

Dentro del archivo ```ej1.2config.json``` de configuracion, se encuentran los siguentes parametros:

- data_source: path
- iterations: integer
- learning_rate: float [0,1]
- constant_learning_rate: [true, false]

## Red de Hopfield
Para la red de Hopfield, ejecutar:

``python3 ej2hopfield.py config/ej2config.json``

Dentro del archivo ```ej2config.json``` de configuracion, se encuentran los siguentes parametros:

- data_source: path
- iterations: integer
- noise: float [0,1]
