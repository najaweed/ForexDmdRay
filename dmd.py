from utils import *


class DynamicMode:
    def __init__(self,
                 ohlc_df: pd.DataFrame,
                 config,
                 ):
        self.df = ohlc_df
        self.config = config

        self.emb_x = time_series_embedding(self.df.close, 1, int(len(self.df) * self.config['embd_ratio']))
        self.X1, self.X2 = self.emb_x[:, :self.emb_x.shape[1] - 1], self.emb_x[:, 1:self.emb_x.shape[1]]

        # self.smooth_x, self.rank_svd = self.svd_smoother()

    def svd_smoother(self):
        u, s, v_h = np.linalg.svd(self.emb_x, full_matrices=False)
        for r in range(1, self.emb_x.shape[1], 1):
            u_r, s_r, v_h_r = u[:, 0:r], s[0:r], v_h[0:r, :]
            xx = np.matmul(np.matmul(u_r, np.diag(s_r)), v_h_r)
            x_df = self.df.iloc[-len(self.emb_x[-1, :]):, :].copy()
            dist = ohlc_distance(x_df, xx[-1, :])
            if dist <= self.config['min_dist_threshold']:
                return xx[-1, :], r
        else:
            return self.emb_x[-1, :], 2

    def recon_dmd(self):
        u, s, v_h = np.linalg.svd(self.X1, full_matrices=False)
        r = self.config['svd_rank']  # self.rank_svd
        u_r, s_r, v_h_r = u[:, 0:r], s[0:r], v_h[0:r, :]
        # A = self.X2 @ v_h_r.conj().T @ np.diag(1 / s_r) @ u_r.conj().T

        a_tilde = u_r.conj().T @ self.X2 @ v_h_r.conj().T @ np.diag(1 / s_r)
        x_lambda, w = np.linalg.eig(a_tilde)

        # compute the Φ matrix
        phi = self.X2 @ v_h_r.conj().T @ np.diag(1 / s_r) @ w
        omega = np.log(x_lambda + 1j * 0)
        b, residuals, rank, s = np.linalg.lstsq(phi, self.X1[:, 0], rcond=None)
        # compute time dynamics
        t_end = self.emb_x.shape[1] + self.config['valid_step'] + self.config['predict_step']
        time_dynamics = np.zeros((len(omega), t_end), dtype=float) + 1j * 0
        t = np.array(range(1, time_dynamics.shape[1] + 1), dtype=float)
        for i in range(0, t_end):
            time_dynamics[:, i] = b * np.exp(omega * t[i])

        # reconstruct training data
        dmd_rec = phi @ time_dynamics
        recon_train = dmd_rec[-1, :self.emb_x.shape[1]].real

        valid = dmd_rec[-1, self.emb_x.shape[1]:self.emb_x.shape[1] + self.config['valid_step']].real
        predict = dmd_rec[-1, self.emb_x.shape[1] + self.config['valid_step']:].real

        return recon_train, valid, predict


class DmdOptModel:
    def __init__(self,
                 step_df,
                 config,
                 ):
        self.df = step_df.copy()
        self.config = config
        self.train, self.valid = split_train_valid(self.df.copy(), 0.1)

        self.config['valid_step'] = len(self.valid)
        self.config['predict_step'] = int(len(self.valid) / 2)
        self.model = DynamicMode(self.train, self.config)

    def forcast(self):
        recon_train, valid, predict = self.model.recon_dmd()

        return recon_train, valid, predict

    def loss_objective(self, ):
        recon_train, model_valid, predict = self.forcast()
        loss_v = ohlc_distance(self.valid, model_valid)
        loss_r = ohlc_distance(self.train.iloc[-len(recon_train):, :], recon_train)
        loss_last = ohlc_distance(self.valid.iloc[-3:, :], model_valid[-3:])
        return (2 * loss_v + loss_r + 2 * loss_last) / 5


# # TEST CLASS
# from test_utils import *
#
# df_input = x_symbol_df
#
#
# def ray_objective(config):
#     x_config = {'min_dist_threshold': 1, 'embd_ratio': config["a"], 'svd_rank': config["b"]}
#     loss = DmdOptModel(step_df=df_input, config=x_config).loss_objective()
#     return {'score': loss}
#
#
# import optuna
#
#
# def objective(trial):
#     # i = trial.suggest_float('min_dist_threshold', 0.2, 2)
#     j = trial.suggest_float('embd_ratio', 0.5, 0.9)
#     k = trial.suggest_int('svd_rank', 1, 100)
#
#     x_config = {'min_dist_threshold': 1, 'embd_ratio': j, 'svd_rank': k}
#     loss = DmdOptModel(step_df=x_symbol_df, config=x_config).loss_objective()
#     return loss
#
#
# #
# study = optuna.create_study()
# study.optimize(objective, n_trials=800, show_progress_bar=True)
# print(study.best_params, study.best_value)
#
# dmd = DmdOptModel(x_symbol_df, study.best_params)
# recon, valid, predict = dmd.forcast()
# plt.plot(recon)
# import mplfinance as mpf
#
# x_df = x_symbol_df.iloc[-(len(recon) + len(valid)):, :]
# # mpf.plot(x_df,type='candle')
# #
# plt.plot(x_symbol_df.close.to_numpy()[-(len(recon) + len(valid)):])
# plt.plot(x_symbol_df.high.to_numpy()[-(len(recon) + len(valid)):])
# plt.plot(x_symbol_df.low.to_numpy()[-(len(recon) + len(valid)):])
#
# plt.plot(range(len(recon), len(recon) + len(valid)), valid)
# plt.plot(range(len(recon) + len(valid), len(recon) + len(predict) + len(valid)), predict)
#
# plt.show()
