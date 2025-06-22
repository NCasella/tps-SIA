from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
import numpy as np

class VariationalAutoencoder:
    def __init__(self,encoder,decoder):
        self.encoder:MultilayerPerceptron =encoder
        self.decoder:MultilayerPerceptron =decoder

    def encode(self, x: np.ndarray):
        """Encode input to latent space parameters (μ and logσ²)"""
        # Get encoder output (size 2*latent_dim)
        h_out,h_partials = self.encoder.feedfoward(MultilayerPerceptron.get_input_with_bias(x))
        
        latent_space_dim=self.decoder.layers_structure[0]

        mu, logvar = h_out[-1].flatten()[:latent_space_dim],h_out[-1].flatten()[latent_space_dim:]
        
        return mu, logvar, h_out, h_partials
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(size=std.shape)
        return mu + eps * std,std,eps

    def kl_divergence(self, mu, log_var):
        return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

    def decode(self, z: np.ndarray) -> np.ndarray:
        x,x_parcials=self.decoder.feedfoward(MultilayerPerceptron.get_input_with_bias(z))
        return x[-1], x, x_parcials  
       
    def forward(self, x: np.ndarray):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(dataset)):
                x = dataset[i].reshape(1, -1)

                # Encoder pass
                mu, log_var, enc_activations, enc_partials = self.encode(x)

                # Sampling z
                z,std,eps = self.reparameterize(mu, log_var)

                # Decoder pass
                x_hat, dec_acts, dec_partials = self.decode(z)

                # Loss calculation
                recon_loss = np.sum((x - x_hat) ** 2)
                kl_loss = self.kl_divergence(mu, log_var)
                loss = recon_loss + kl_loss
                total_loss += loss

                # Decoder backpropagation (reconstruction only)
                err_out = (x_hat - x)
                self.decoder.backpropagate(dec_partials, err_out, dec_acts)

                delta = err_out * self.decoder.calculate_derivate(dec_partials[-1])
                for l in range(len(self.decoder.weights) - 1, 0, -1):
                    Wn = self.decoder.weights[l][:-1, :]
                    delta = (delta @ Wn.T) * self.decoder.calculate_derivate(dec_partials[l - 1])

                W0 = self.decoder.weights[0]
                d_input = delta @ W0.T
                dz = d_input[:, :self.decoder.layers_structure[0]]

                # Combine with KL gradients
                dz_mu = dz + mu
                dz_logvar = dz * eps * std * 0.5 + 0.5 * (np.exp(log_var) - 1)

                dz_combined = np.hstack([dz_mu, dz_logvar]).reshape(1, -1)

                self.encoder.backpropagate(enc_partials, dz_combined, enc_activations)

            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")