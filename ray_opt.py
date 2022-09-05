from dmd import DmdOptModel
import pandas as pd

df_input = pd.read_csv('data.csv',index_col='time')
def ray_objective(config):
    x_config = {'min_dist_threshold': 1, 'embd_ratio': config["a"], 'svd_rank': config["b"]}
    loss = DmdOptModel(step_df=df_input, config=x_config).loss_objective()
    return {'score': loss}