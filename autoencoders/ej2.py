import json
import os
import sys
from PIL import Image
import numpy as np
from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
from src.perceptrons.sigmoid_functions import get_sigmoid_function_and_derivate
from src.utils import png_to_rgba_array
from src.variational_autoencoder import VariationalAutoencoder
from src.perceptrons.optimizers.optimizers import *

if __name__=="__main__":

    with open(sys.argv[1],"r") as f:
        config=json.load(f)

    input_directory = config["input_directory"]
    files = os.listdir(input_directory)

    png_files = []

    for file in files:
        file_path = os.path.join(input_directory,file)
        if os.path.isfile(file_path) and file.endswith(".png"):
            png_files.append(file_path)

    encode_layers_config=config["encoder_layers"]
    decode_layers_config=config["decoder_layers"]
    learning_rate=config["learning_rate"]
    function=config["activation_function"]
    f,df=get_sigmoid_function_and_derivate(1,function)
    epochs=config["epochs"]
    epsilon=config["epsilon"]
    optimizer_value=config["optimizer"]
    optimizer_alpha=config["optimizer_alpha"]
    optimizer_beta1=config["optimizer_beta_1"]
    optimizer_beta2=config["optimizer_beta_2"]
    optimizer_epsilon=config["optimizer_epsilon"]
    

    enc_input_size,dec_input_size = encode_layers_config[0],decode_layers_config[0]
    encode_layer_shapes = []
    decode_layer_shapes=[]
    enc_current_size = enc_input_size + 1
    dec_current_size=dec_input_size +1

    for enc_layer_size,dec_layer_size in zip(encode_layers_config[1:],decode_layers_config[1:]):
        encode_layer_shapes.append((enc_current_size, enc_layer_size))
        decode_layer_shapes.append((dec_current_size, dec_layer_size))
        enc_current_size=enc_layer_size + 1
        dec_current_size=dec_layer_size + 1
    
    enc_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, encode_layer_shapes)
    dec_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, decode_layer_shapes)



    encoder=MultilayerPerceptron(learning_rate,f,df,encode_layers_config,enc_opt)
    decoder=MultilayerPerceptron(learning_rate,f,df,decode_layers_config,dec_opt)

    input=[png_to_rgba_array(file) for file in png_files]
    print(input)

    vartiational_autoencoder: VariationalAutoencoder=VariationalAutoencoder(encoder,decoder)
    vartiational_autoencoder.train(input,epochs)
    num_generate = 5
    generated_samples = vartiational_autoencoder.generate(num_generate)  # shape (num_generate, data_dim)

    # 4. Guardar las muestras generadas en archivos PNG
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(generated_samples):
        # sample puede venir como vector plano, por ejemplo shape (16*16*4,) = (1024,)
        # Lo redimensionamos a (16,16,4)
        img_array = sample.reshape((16, 16, 4))

        # Si los valores están en float (ej: 0-1), convertimos a uint8 0-255
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        # Crear imagen PIL
        img = Image.fromarray(img_array, mode='RGBA')

        # Guardar
        filename = os.path.join(output_dir, f"generated_{i}.png")
        img.save(filename)
        print(f"Saved {filename}")



