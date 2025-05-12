from os import path
import pandas as pd
import numpy as np

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer

path_train = "../df_train.parquet"
df_train = pd.read_parquet(path_train)

df_train["carb"] = df_train["carb"].fillna(0)
df_train["bolus"] = df_train["bolus"].fillna(0)
df_train["basal_rate"] = df_train["basal_rate"].ffill().bfill()

df_timegan = df_train[['bolus', 'basal_rate', 'carb', 'Value']]
cols = list(df_timegan.columns)

seq_len = 6
n_seq = 4
hidden_dim = 24
gamma = 1

noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
learning_rate = 5e-4
epochs = 10

gan_args = ModelParameters(
    batch_size=batch_size, lr=learning_rate, noise_dim=noise_dim, layers_dim=dim
)

train_args = TrainParameters(
    epochs=epochs, sequence_length=seq_len, number_sequences=n_seq
)

synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
synth.fit(df_timegan, train_args, num_cols=cols)
synth.save("model_timegan.pkl")
