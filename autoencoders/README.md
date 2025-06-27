# tp5-SIA_Autoencoder

Todas las dependencias necesarias estan dentro del archivo ```requirements.txt```, que se pueden instalar con:

``pip install -r requirements.txt``

## Autoencoder/DAE
Para el autoencoder, ejecutar:
``python3 ej1a.py config/ej1config.json``

Para el denoising autoencoder: 
``python3 ej1b.py config/ej1config.json``

Dentro del archivo ```ej1config.json``` de configuracion, se encuentran los siguentes parametros modificables:

- epochs: integer
- epsilon: integer
- learning_rate: float [0,1]
- layers: [integer]
- similariry_metric: ["relu","tanh","logistic","softplus"]
- font: {1,2,3}
- optimizer: ["adam","sgd","momentum"]
- standard_deviation: float [0,1]


## Variational Autoencoder

Para el VAE, ejecutar:
``python3 ej2.py config/ej2config.json``

Dentro del archivo ```ej2config.json``` de configuracion, se encuentra el siguiente parametros extra, ademas de los anteriores:

- input_directory: Path

