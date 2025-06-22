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

                # Sample z
                z, std, eps = self.reparameterize(mu, log_var)

                # Decoder pass
                x_hat, dec_activations, dec_partials = self.decode(z)

                # Compute losses
                recon_loss = np.sum((x - x_hat) ** 2)
                kl_loss = self.kl_divergence(mu, log_var)
                loss = recon_loss + kl_loss
                total_loss += loss

                # Decoder backpropagation: reconstruction error gradient
                err_out = x_hat - x  # gradient of reconstruction loss wrt output
                self.decoder.backpropagate(dec_partials, err_out, dec_activations)

                # Gradient at latent space z from decoder backprop
                delta = err_out * self.decoder.calculate_derivate(dec_partials[-1])
                for l in range(len(self.decoder.weights) - 2, -1, -1):
                    Wn = self.decoder.weights[l + 1][:-1, :]
                    delta = (delta @ Wn.T) * self.decoder.calculate_derivate(dec_partials[l])

                # Gradient w.r.t input to decoder (latent vector z)
                d_input = delta @ self.decoder.weights[0][:-1, :].T

                # Now split gradients for mu and logvar for encoder backpropagation

                # Gradients from reconstruction error via z
                dz_mu = d_input  # since dz/dmu = 1

                # dz/dlogvar = 0.5 * eps * std (chain rule)
                dz_logvar = d_input * eps * std * 0.5

                # Gradients from KL divergence term:
                # dKL/dmu = mu
                # dKL/dlogvar = 0.5 * (exp(logvar) - 1)
                dkl_mu = mu
                dkl_logvar = 0.5 * (np.exp(log_var) - 1)

                # Combine gradients
                grad_mu = dz_mu + dkl_mu
                grad_logvar = dz_logvar + dkl_logvar

                dz_combined = np.hstack([grad_mu, grad_logvar]).reshape(1, -1)

                # Encoder backpropagation
                self.encoder.backpropagate(enc_partials, dz_combined, enc_activations)

            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    def generate(self, num_samples=1):
        """
        Genera nuevas muestras a partir del decoder del VAE.

        Args:
            num_samples: cantidad de muestras a generar.

        Returns:
            samples: numpy array con las muestras generadas.
        """
        latent_dim = self.decoder.layers_structure[0]
        z = np.random.normal(size=(num_samples, latent_dim))
        generated, _, _ = self.decode(z)
        return generated