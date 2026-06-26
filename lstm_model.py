"""
Implementasi LSTM dari nol menggunakan NumPy (tanpa TensorFlow/PyTorch),
karena environment build ini tidak memiliki akses internet untuk instalasi
TensorFlow. Implementasi ini mencakup forward pass LSTM penuh (4 gate),
backpropagation through time (BPTT), dan optimizer Adam.

Pada komputer pengguna sendiri, modul ini bisa diganti dengan implementasi
TensorFlow/Keras (lihat fungsi `try_import_keras_lstm` di bawah) tanpa
mengubah API yang dipakai oleh aplikasi Tkinter.
"""

import numpy as np


def try_import_keras_lstm():
    """Coba pakai TensorFlow/Keras jika tersedia di komputer pengguna."""
    try:
        import tensorflow as tf  # noqa
        return True
    except ImportError:
        return False


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class NumpyLSTM:
    """
    LSTM many-to-one untuk forecasting time series, dilatih dengan Adam.

    Input  : X shape (n_samples, seq_len, n_features)
    Output : y shape (n_samples, 1)  -> prediksi 1 langkah ke depan
    """

    def __init__(self, n_features, n_hidden=32, dropout=0.0, seed=42):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.dropout = dropout
        rng = np.random.RandomState(seed)
        H, D = n_hidden, n_features
        scale = 1.0 / np.sqrt(H + D)

        def init(shape):
            return (rng.rand(*shape) * 2 - 1) * scale

        # Gabungan input: [x_t, h_{t-1}] -> dim (D+H)
        self.Wf = init((D + H, H)); self.bf = np.zeros((1, H))
        self.Wi = init((D + H, H)); self.bi = np.zeros((1, H))
        self.Wc = init((D + H, H)); self.bc = np.zeros((1, H))
        self.Wo = init((D + H, H)); self.bo = np.zeros((1, H))
        self.Wy = init((H, 1));     self.by = np.zeros((1, 1))

        self.params = [self.Wf, self.bf, self.Wi, self.bi,
                       self.Wc, self.bc, self.Wo, self.bo,
                       self.Wy, self.by]
        # Adam moments
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

        self.history = {"loss": [], "val_loss": []}

    # ---------- forward ----------
    def _forward(self, X, train=True):
        N, T, D = X.shape
        H = self.n_hidden
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        cache = []
        drop_masks = []
        for t in range(T):
            xt = X[:, t, :]
            if train and self.dropout > 0:
                mask = (np.random.rand(N, D) > self.dropout).astype(np.float64)
                xt_used = xt * mask
                drop_masks.append(mask)
            else:
                xt_used = xt
                drop_masks.append(None)
            z = np.concatenate([xt_used, h], axis=1)
            f = sigmoid(z @ self.Wf + self.bf)
            i = sigmoid(z @ self.Wi + self.bi)
            g = tanh(z @ self.Wc + self.bc)
            o = sigmoid(z @ self.Wo + self.bo)
            c = f * c + i * g
            h = o * tanh(c)
            cache.append((z, f, i, g, o, c.copy(), h.copy()))
        y_pred = h @ self.Wy + self.by
        return y_pred, cache, drop_masks

    def predict(self, X, batch_size=256):
        preds = []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            yp, _, _ = self._forward(xb, train=False)
            preds.append(yp)
        return np.concatenate(preds, axis=0)

    # ---------- backward (BPTT) ----------
    def _backward(self, X, y_true, y_pred, cache, drop_masks):
        N, T, D = X.shape
        H = self.n_hidden
        dWy = cache[-1][6].T @ (2 * (y_pred - y_true) / N)
        dby = np.sum(2 * (y_pred - y_true) / N, axis=0, keepdims=True)

        dh_next = (2 * (y_pred - y_true) / N) @ self.Wy.T
        dc_next = np.zeros((N, H))

        grads = {k: np.zeros_like(getattr(self, k)) for k in
                  ["Wf", "bf", "Wi", "bi", "Wc", "bc", "Wo", "bo"]}

        c_prev_list = [np.zeros((N, H))] + [cache[t][5] for t in range(T - 1)]

        for t in reversed(range(T)):
            z, f, i, g, o, c_t, h_t = cache[t]
            c_prev = c_prev_list[t]

            dh = dh_next
            do = dh * tanh(c_t)
            do_raw = do * o * (1 - o)

            dc = dc_next + dh * o * (1 - tanh(c_t) ** 2)
            di = dc * g
            di_raw = di * i * (1 - i)
            dg = dc * i
            dg_raw = dg * (1 - g ** 2)
            df = dc * c_prev
            df_raw = df * f * (1 - f)

            grads["Wf"] += z.T @ df_raw; grads["bf"] += df_raw.sum(0, keepdims=True)
            grads["Wi"] += z.T @ di_raw; grads["bi"] += di_raw.sum(0, keepdims=True)
            grads["Wc"] += z.T @ dg_raw; grads["bc"] += dg_raw.sum(0, keepdims=True)
            grads["Wo"] += z.T @ do_raw; grads["bo"] += do_raw.sum(0, keepdims=True)

            dz = (df_raw @ self.Wf.T + di_raw @ self.Wi.T +
                  dg_raw @ self.Wc.T + do_raw @ self.Wo.T)
            dh_next = dz[:, D:]
            dc_next = dc * f

        for k in grads:
            grads[k] = np.clip(grads[k], -5, 5)
        return grads, dWy, dby

    def _adam_step(self, grads, dWy, dby, lr):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        all_grads = [grads["Wf"], grads["bf"], grads["Wi"], grads["bi"],
                     grads["Wc"], grads["bc"], grads["Wo"], grads["bo"],
                     dWy, dby]
        for idx, (p, g) in enumerate(zip(self.params, all_grads)):
            self.m[idx] = beta1 * self.m[idx] + (1 - beta1) * g
            self.v[idx] = beta2 * self.v[idx] + (1 - beta2) * (g ** 2)
            mhat = self.m[idx] / (1 - beta1 ** self.t)
            vhat = self.v[idx] / (1 - beta2 ** self.t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, y, X_val=None, y_val=None, epochs=20, batch_size=32,
            lr=0.001, progress_callback=None, stop_flag=None):
        n = len(X)
        for ep in range(epochs):
            if stop_flag is not None and stop_flag():
                break
            idx = np.random.permutation(n)
            X_sh, y_sh = X[idx], y[idx]
            epoch_loss = 0.0
            n_batches = 0
            for s in range(0, n, batch_size):
                if stop_flag is not None and stop_flag():
                    break
                xb = X_sh[s:s + batch_size]
                yb = y_sh[s:s + batch_size]
                if len(xb) == 0:
                    continue
                y_pred, cache, drop_masks = self._forward(xb, train=True)
                loss = np.mean((y_pred - yb) ** 2)
                grads, dWy, dby = self._backward(xb, yb, y_pred, cache, drop_masks)
                self._adam_step(grads, dWy, dby, lr)
                epoch_loss += loss
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            self.history["loss"].append(train_loss)
            if X_val is not None and len(X_val) > 0:
                val_pred = self.predict(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
            else:
                val_loss = train_loss
            self.history["val_loss"].append(val_loss)
            if progress_callback:
                progress_callback(ep + 1, epochs, train_loss, val_loss)
        return self.history
