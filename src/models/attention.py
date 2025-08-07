import tensorflow as tf
from keras.layers import Input, LSTM, Dense, RepeatVector, Concatenate, Dropout, Activation, Dot, Flatten, Reshape
from keras.models import Model
from keras.activations import softmax
import matplotlib.pyplot as plt


def softmax_axis1(x):
    return softmax(x, axis=1)


def temporal_attention(a, s_prev, Tx, h_s):
    t_repeator = RepeatVector(Tx)
    t_densor = Dense(1, activation="relu")
    concatenator = Concatenate(axis=-1)
    activator = Activation(softmax_axis1)
    dotor = Dot(axes=1)

    s_prev = t_repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = t_densor(concat)
    alphas = activator(e)
    context = dotor([alphas, a])
    return context, alphas


def spatial_attention(v, s_prev, inp_var, h_s):
    s_repeator = RepeatVector(inp_var)
    s_densor_1 = Dense(h_s, activation="relu")
    s_densor_2 = Dense(1, activation="relu")
    concatenator = Concatenate(axis=-1)
    activator = Activation(softmax_axis1)
    dotor = Dot(axes=1)

    s_fc = s_densor_1(v)
    s_prev = s_repeator(s_prev)
    concat = concatenator([s_fc, s_prev])
    e = s_densor_2(concat)
    betas = activator(e)
    context = dotor([betas, s_fc])
    return context, betas


def build_attention_model(Tx, Ty, inp_var, h_s=32, dropout=0.2, con_dim=4):
    encoder_input = Input(shape=(Tx, inp_var))
    spatial_input = Input(shape=(inp_var, Tx))
    s0 = Input(shape=(h_s,))
    c0 = Input(shape=(h_s,))
    yhat0 = Input(shape=(1,))

    ts, tc = s0, c0
    ss, sc = s0, c0
    yhat = yhat0

    outputs = []
    alphas_betas_list = []

    lstm_1, _, _ = LSTM(h_s, return_state=True, return_sequences=True)(encoder_input)
    lstm_1 = Dropout(dropout)(lstm_1)
    lstm_2, _, _ = LSTM(h_s, return_state=True, return_sequences=True)(lstm_1)
    lstm_2 = Dropout(dropout)(lstm_2)

    t_decoder_lstm = LSTM(h_s, return_state=True)
    s_decoder_lstm = LSTM(h_s, return_state=True)
    flatten = Flatten()
    concatenator = Concatenate(axis=-1)

    for t in range(Ty):
        t_context, alphas = temporal_attention(lstm_2, ts, Tx, h_s)
        t_context = Dense(con_dim, activation="relu")(t_context)
        t_context = flatten(t_context)
        t_context = concatenator([t_context, yhat])
        t_context = Reshape((1, con_dim + 1))(t_context)

        s_context, betas = spatial_attention(spatial_input, ss, inp_var, h_s)
        s_context = Dense(con_dim, activation="relu")(s_context)
        s_context = flatten(s_context)
        s_context = concatenator([s_context, yhat])
        s_context = Reshape((1, con_dim + 1))(s_context)

        ts, _, tc = t_decoder_lstm(t_context, initial_state=[ts, tc])
        ts = Dropout(dropout)(ts)
        ss, _, sc = s_decoder_lstm(s_context, initial_state=[ss, sc])
        ss = Dropout(dropout)(ss)

        context = concatenator([ts, ss])
        yhat = Dense(1, activation="linear")(context)

        outputs.append(yhat)
        alphas_betas_list += [yhat, alphas, betas]

    pred_model = Model([encoder_input, spatial_input, s0, c0, yhat0], outputs)
    prob_model = Model([encoder_input, spatial_input, s0, c0, yhat0], alphas_betas_list)

    return pred_model, prob_model


def plot_attention_weights(attn_outputs, Tx, inp_var, sample_idx=0):
    alphas = attn_outputs[1]  # temporal
    betas = attn_outputs[2]  # spatial

    plt.figure()
    plt.title("Temporal Attention")
    plt.stem(range(Tx), alphas[sample_idx, :, 0])
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("Spatial Attention")
    plt.bar(range(inp_var), betas[sample_idx, :, 0])
    plt.xlabel("Feature Index")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.show()

def run_attention_pipeline(csv_path="data/eq_catalog.csv", Tx=10, Ty=1, h_s=32):
    from src.preprocessing.load_catalog import load_catalog
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # --- Load and scale ---
    df, _ = load_catalog(csv_path)
    features = df[['origin_time', 'latitude', 'longitude', 'depth', 'magnitude']].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # --- Sequence Processing ---
    X, y = [], []
    for i in range(Tx, len(scaled)):
        X.append(scaled[i - Tx:i])
        y.append(scaled[i, -1])
    X = np.array(X)
    y = np.array(y).reshape(-1, Ty, 1)
    s = X.transpose(0, 2, 1)

    inp_var = X.shape[2]
    pred_model, prob_model = build_attention_model(Tx, Ty, inp_var, h_s=h_s)

    # --- Init LSTM State ---
    s0 = np.zeros((X.shape[0], h_s))
    c0 = np.zeros((X.shape[0], h_s))
    yhat0 = np.zeros((X.shape[0], 1))

    # --- Predict and Plot ---
    attn_outputs = prob_model.predict([X, s, s0, c0, yhat0])
    plot_attention_weights(attn_outputs, Tx, inp_var)