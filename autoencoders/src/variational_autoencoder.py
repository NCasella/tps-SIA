import numpy as np

class VariationalAutoencoder:
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: input -> hidden -> (mu || logvar)
        self.enc_w1 = np.random.randn(input_dim + 1, hidden_dim) * 0.01
        self.enc_w2 = np.random.randn(hidden_dim + 1, latent_dim * 2) * 0.01

        # Decoder: latent -> hidden -> output
        self.dec_w1 = np.random.randn(latent_dim + 1, hidden_dim) * 0.01
        self.dec_w2 = np.random.randn(hidden_dim + 1, input_dim) * 0.01

        self.lr = 0.001

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def add_bias(self, x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    def encode(self, x):
        x = self.add_bias(x)
        h1 = self.relu(x @ self.enc_w1)
        h1_b = self.add_bias(h1)
        h2 = h1_b @ self.enc_w2
        mu = h2[:, :self.latent_dim]
        logvar = h2[:, self.latent_dim:]
        return mu, logvar, (x, h1, h1_b)

    def decode(self, z):
        z = self.add_bias(z)
        h1 = self.relu(z @ self.dec_w1)
        h1_b = self.add_bias(h1)
        x_hat = self.sigmoid(h1_b @ self.dec_w2)
        return x_hat, (z, h1, h1_b)

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std, std, eps

    def kl_divergence(self, mu, logvar):
        return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

    def forward(self, x):
        mu, logvar, enc_cache = self.encode(x)
        z, std, eps = self.reparameterize(mu, logvar)
        x_hat, dec_cache = self.decode(z)
        return x_hat, mu, logvar, enc_cache, dec_cache, z, std, eps

    def backward(self, x, x_hat, mu, logvar, enc_cache, dec_cache, z, std, eps):
        # Decoder backprop
        dz, dh1, dh1_b = dec_cache
        dz = self.add_bias(z)

        dL_dxhat = (x_hat - x)  # MSE gradient
        dxhat_dz2 = self.sigmoid_deriv(hh := dz @ self.dec_w1)
        grad_dec_w2 = dh1_b.T @ dL_dxhat
        dh1_no_bias = dL_dxhat @ self.dec_w2[:-1].T * self.relu_deriv(h1 := dh1_b[:, :-1])

        grad_dec_w1 = dz.T @ dh1_no_bias

        # Backprop through reparameterization
        dz = dh1_no_bias @ self.dec_w1[:-1].T  # dz/dL

        dmu = dz + mu
        dlogvar = dz * eps * std * 0.5 + 0.5 * (np.exp(logvar) - 1)

        x0, h1, h1_b = enc_cache
        h1_b = self.add_bias(h1)

        dconcat = np.hstack([dmu, dlogvar])
        grad_enc_w2 = h1_b.T @ dconcat
        dh1_enc = dconcat @ self.enc_w2[:-1].T * self.relu_deriv(h1)

        grad_enc_w1 = x0.T @ dh1_enc

        # Update weights
        self.dec_w2 -= self.lr * grad_dec_w2
        self.dec_w1 -= self.lr * grad_dec_w1
        self.enc_w2 -= self.lr * grad_enc_w2
        self.enc_w1 -= self.lr * grad_enc_w1

    def train(self, dataset, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for x in dataset:
                x = x.reshape(1, -1)
                x_hat, mu, logvar, enc_cache, dec_cache, z, std, eps = self.forward(x)
                recon_loss = np.mean((x - x_hat) ** 2)
                kl = self.kl_divergence(mu, logvar)
                loss = recon_loss + kl
                self.backward(x, x_hat, mu, logvar, enc_cache, dec_cache, z, std, eps)
                total_loss += loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def generate(self, num_samples=1):
        z = np.random.randn(num_samples, self.latent_dim)
        x_hat, _ = self.decode(z)
        return x_hat

    def predict(self, x):
        x = x.reshape(1, -1)
        mu, logvar, _ = self.encode(x)
        z, _, _ = self.reparameterize(mu, logvar)
        x_hat, _ = self.decode(z)
        return x_hat
