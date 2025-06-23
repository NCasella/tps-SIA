import numpy as np

class VariationalAutoencoder:
    def __init__(self, encoder_layers, decoder_layers, activation_function, activation_derivative, learning_rate=0.001, optimizer=None):
        """
        encoder_layers: List[int] e.g. [input_dim, ..., 2*latent_dim]
                        The last layer size must be 2*latent_dim (mu and logvar)
        decoder_layers: List[int] e.g. [latent_dim, ..., output_dim]
                        The first layer size must be latent_dim
        activation_function: callable (e.g. relu or sigmoid)
        activation_derivative: callable, derivative of activation_function
        learning_rate: float
        """
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.act = activation_function
        self.act_deriv = activation_derivative
        self.lr = learning_rate
        self.optimizer = optimizer

        self.enc_weights = [np.random.randn(in_dim + 1, out_dim) * 0.1
                            for in_dim, out_dim in zip(encoder_layers[:-1], encoder_layers[1:])]
        self.dec_weights = [np.random.randn(in_dim + 1, out_dim) * 0.1
                            for in_dim, out_dim in zip(decoder_layers[:-1], decoder_layers[1:])]

        self.latent_dim = decoder_layers[0]

    def add_bias(self, x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    def mlp_forward(self, x, weights):
        activations = [x]
        pre_activations = []
        for W in weights:
            x_b = self.add_bias(activations[-1])
            z = x_b @ W
            pre_activations.append(z)
            x = self.act(z)
            activations.append(x)
        return activations, pre_activations

    def mlp_backward(self, weights, activations, pre_activations, grad_output):
        grads = [None] * len(weights)
        delta = grad_output
        for i in reversed(range(len(weights))):
            act_deriv = self.act_deriv(pre_activations[i])
            delta = delta * act_deriv  # element-wise multiply
            x_b = self.add_bias(activations[i])
            grads[i] = x_b.T @ delta
            if i > 0:
                W_no_bias = weights[i][:-1, :]
                delta = delta @ W_no_bias.T
        return grads

    def encode(self, x):
        activations, pre_activations = self.mlp_forward(x, self.enc_weights)
        output = activations[-1]
        mu = output[:, :self.latent_dim]
        logvar = output[:, self.latent_dim:]
        return mu, logvar, activations, pre_activations

    def decode(self, z):
        activations, pre_activations = self.mlp_forward(z, self.dec_weights)
        return activations[-1], activations, pre_activations

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std, std, eps

    def kl_divergence(self, mu, logvar):
        return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

    def forward(self, x):
        mu, logvar, enc_acts, enc_preacts = self.encode(x)
        z, std, eps = self.reparameterize(mu, logvar)
        x_hat, dec_acts, dec_preacts = self.decode(z)
        return x_hat, mu, logvar, enc_acts, enc_preacts, dec_acts, dec_preacts, z, std, eps

    def backward(self, x, x_hat, mu, logvar, enc_acts, enc_preacts, dec_acts, dec_preacts, z, std, eps):
        d_x_hat = 2 * (x_hat - x) / x.shape[0]

        dec_grads = self.mlp_backward(self.dec_weights, dec_acts, dec_preacts, d_x_hat)

        delta = d_x_hat
        dz = None

        for i in reversed(range(len(self.dec_weights))):
            act_deriv = self.act_deriv(dec_preacts[i])
            delta = delta * act_deriv
            x_b = self.add_bias(dec_acts[i])
            if i == 0:
                W_no_bias = self.dec_weights[i][:-1, :]
                dz = delta @ W_no_bias.T
            delta = delta @ self.dec_weights[i][:-1, :].T

        latent_dim = self.latent_dim
        dz_mu = dz
        dz_logvar = dz * eps * std * 0.5

        dkl_mu = mu
        dkl_logvar = 0.5 * (np.exp(logvar) - 1)

        grad_mu = dz_mu + dkl_mu
        grad_logvar = dz_logvar + dkl_logvar

        grad_combined = np.hstack([grad_mu, grad_logvar])

        enc_grads = self.mlp_backward(self.enc_weights, enc_acts, enc_preacts, grad_combined)

        if self.optimizer is not None:
            self.enc_weights = self.optimizer(self.enc_weights, enc_grads)
            self.dec_weights = self.optimizer(self.dec_weights, dec_grads)
        else:
            for i in range(len(self.enc_weights)):
                self.enc_weights[i] -= self.lr * enc_grads[i]
            for i in range(len(self.dec_weights)):
                self.dec_weights[i] -= self.lr * dec_grads[i]

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for x in dataset:
                x = x.reshape(1, -1)
                x_hat, mu, logvar, enc_acts, enc_preacts, dec_acts, dec_preacts, z, std, eps = self.forward(x)
                recon_loss = np.mean((x - x_hat) ** 2)
                kl = self.kl_divergence(mu, logvar)
                loss = recon_loss + kl
                total_loss += loss
                self.backward(x, x_hat, mu, logvar, enc_acts, enc_preacts, dec_acts, dec_preacts, z, std, eps)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def generate(self, num_samples=1):
        z = np.random.randn(num_samples, self.latent_dim)
        activations, _ = self.mlp_forward(z, self.dec_weights)
        return activations[-1]

    def predict(self, x):
        mu, logvar, _, _ = self.encode(x)
        z, _, _ = self.reparameterize(mu, logvar)
        x_hat, _ = self.mlp_forward(z, self.dec_weights)
        return x_hat
