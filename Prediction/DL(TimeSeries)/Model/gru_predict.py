import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Embedding, GRU, Dense, Dropout, Concatenate, TimeDistributed,
                                     Activation, Flatten, RepeatVector, Permute, Multiply, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# --------------------- Helper Functions ---------------------

@tf.keras.utils.register_keras_serializable()
def reduce_sum_axis1(x):
    return tf.reduce_sum(x, axis=1)

def create_dataset(data, time_step=1, include_inter_region=False):
    """
    Create sliding window dataset for each country.
    If include_inter_region is True, the feature order is:
      [country_enc, region_enc, subregion_enc, inter_region_enc, year_norm,
       life_expectancy_women, life_expectancy_men, life_expectancy_women_diff, life_expectancy_men_diff]
    Otherwise, the order is:
      [country_enc, region_enc, subregion_enc, year_norm,
       life_expectancy_women, life_expectancy_men, life_expectancy_women_diff, life_expectancy_men_diff]
    """
    X, y = [], []
    # Determine the columns to use
    if include_inter_region:
        cols = ['country_enc', 'region_enc', 'subregion_enc', 'inter_region_enc', 'year_norm',
                'life_expectancy_women', 'life_expectancy_men',
                'life_expectancy_women_diff', 'life_expectancy_men_diff']
    else:
        cols = ['country_enc', 'region_enc', 'subregion_enc', 'year_norm',
                'life_expectancy_women', 'life_expectancy_men',
                'life_expectancy_women_diff', 'life_expectancy_men_diff']

    # Process each country
    for country in data['country_name'].unique():
        country_data = data[data['country_name'] == country].sort_values(by='year')
        feat = country_data[cols].values
        for i in range(len(feat) - time_step):
            X.append(feat[i:(i + time_step)])
            # Target: next time step's life expectancies (women, men)
            # The last 4 columns are [life_expectancy_women, life_expectancy_men, diff_women, diff_men]
            # 这里取 -4:-2 表示只取 [life_expectancy_women, life_expectancy_men]
            y.append(feat[i + time_step, -4:-2])
    return np.array(X), np.array(y)

def build_model(time_steps, num_country, num_region, num_subregion,
                emb_dim_country=16, emb_dim_region=8, emb_dim_subregion=8,
                num_inter=None, emb_dim_inter=8, include_inter_region=False):
    """
    Build GRU model with optional inter_region feature.
    If include_inter_region is True, an extra input and embedding is used for inter_region.
    Continuous features are assumed to have 5 columns: [year_norm, life_exp_women, life_exp_men, diff_women, diff_men]
    """
    # Input layers for categorical features
    input_country = Input(shape=(time_steps,), name='country_input')
    input_region = Input(shape=(time_steps,), name='region_input')
    input_subregion = Input(shape=(time_steps,), name='subregion_input')
    
    if include_inter_region:
        input_inter = Input(shape=(time_steps,), name='inter_region_input')
    
    # Input layer for continuous features (always 5 features)
    input_cont = Input(shape=(time_steps, 5), name='continuous_input')

    # Embedding layers
    emb_country = Embedding(input_dim=num_country, output_dim=emb_dim_country, input_length=time_steps)(input_country)
    emb_region = Embedding(input_dim=num_region, output_dim=emb_dim_region, input_length=time_steps)(input_region)
    emb_subregion = Embedding(input_dim=num_subregion, output_dim=emb_dim_subregion, input_length=time_steps)(input_subregion)
    
    if include_inter_region:
        emb_inter = Embedding(input_dim=num_inter, output_dim=emb_dim_inter, input_length=time_steps)(input_inter)

    # Project continuous features
    proj_cont = TimeDistributed(Dense(8, activation='relu'))(input_cont)

    # Concatenate all features
    if include_inter_region:
        merged = Concatenate(axis=-1)([emb_country, emb_region, emb_subregion, emb_inter, proj_cont])
    else:
        merged = Concatenate(axis=-1)([emb_country, emb_region, emb_subregion, proj_cont])
    # merged shape: (batch, time_steps, feature_dim)

    # GRU layers
    x = GRU(units=128, return_sequences=True)(merged)
    x = Dropout(0.3)(x)
    x = GRU(units=64, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)   # shape: (batch, time_steps, 1)
    attention = Flatten()(attention)             # shape: (batch, time_steps)
    attention = Activation('softmax')(attention) # shape: (batch, time_steps)
    attention = RepeatVector(64)(attention)      # repeat for 64 units
    attention = Permute([2, 1])(attention)       # shape: (batch, time_steps, 64)
    x = Multiply()([x, attention])

    # Sum over time steps
    x = Lambda(reduce_sum_axis1, output_shape=(64,))(x)

    # Fully connected layers for prediction
    x = Dense(32, activation='relu')(x)
    output_women = Dense(1, activation='linear', name='output_women')(x)
    output_men = Dense(1, activation='linear', name='output_men')(x)

    # Build model
    if include_inter_region:
        model = Model(inputs=[input_country, input_region, input_subregion, input_inter, input_cont],
                      outputs=[output_women, output_men])
    else:
        model = Model(inputs=[input_country, input_region, input_subregion, input_cont],
                      outputs=[output_women, output_men])

    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def compute_metrics(true_vals, pred_vals):
    mse = mean_squared_error(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    return mse, mae, r2

def run_experiment(experiment_type, time_step=5):
    """
    Run one experiment based on the experiment_type:
      - 'baseline': use features [country_name, region, sub-region, year] (no inter-region)
      - 'inter_unknown': include 'inter-region' with missing values filled as 'unknown'
      - 'inter_subregion': include 'inter-region' with missing values filled using sub-region
    Returns:
      1) a dictionary with metrics for both predictions (with_diff / without_diff).
      2) the training history (history.history) for plotting.
    """
    print(f"\nRunning experiment: {experiment_type}")

    # Load dataset
    if experiment_type == 'baseline':
        data = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')
        data = data[['country_name', 'region', 'sub-region', 'year', 'life_expectancy_women', 'life_expectancy_men']]
        include_inter = False
    else:
        data = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')
        data = data[['country_name', 'region', 'sub-region', 'intermediate-region', 'year', 'life_expectancy_women', 'life_expectancy_men']]
        include_inter = True
        # Fill missing values
        if experiment_type == 'inter_unknown':
            data['intermediate-region'] = data['intermediate-region'].fillna('unknown')
        elif experiment_type == 'inter_subregion':
            data['intermediate-region'] = data['intermediate-region'].fillna(data['sub-region'])

    # Data preprocessing
    data['life_expectancy_women'] = pd.to_numeric(data['life_expectancy_women'].astype(str).str.replace(',', '.'), errors='coerce')
    data['life_expectancy_men']   = pd.to_numeric(data['life_expectancy_men'].astype(str).str.replace(',', '.'), errors='coerce')
    data['year'] = data['year'].astype(int)
    
    # Normalize 'year'
    data['year_norm'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[['year']])
    data.sort_values(by=['country_name', 'year'], inplace=True)

    # Label encoding for categorical features
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_subregion = LabelEncoder()
    data['country_enc'] = le_country.fit_transform(data['country_name'].astype(str))
    data['region_enc']   = le_region.fit_transform(data['region'].astype(str))
    data['subregion_enc'] = le_subregion.fit_transform(data['sub-region'].astype(str))

    if include_inter:
        le_inter = LabelEncoder()
        data['inter_region_enc'] = le_inter.fit_transform(data['intermediate-region'].astype(str))

    # Normalize life expectancy values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['life_expectancy_women', 'life_expectancy_men']] = scaler.fit_transform(
        data[['life_expectancy_women', 'life_expectancy_men']]
    )

    # Compute first order differences
    data['life_expectancy_women_diff'] = data.groupby('country_name')['life_expectancy_women'].diff()
    data['life_expectancy_men_diff']   = data.groupby('country_name')['life_expectancy_men'].diff()
    data = data.dropna(subset=['life_expectancy_women_diff', 'life_expectancy_men_diff'])

    # Create sliding window dataset
    X, y = create_dataset(data, time_step, include_inter_region=include_inter)

    # Split X into categorical and continuous parts
    if include_inter:
        # [0:4] -> [country_enc, region_enc, subregion_enc, inter_region_enc]
        # [4:9] -> [year_norm, life_exp_women, life_exp_men, life_exp_women_diff, life_exp_men_diff]
        X_cat = X[:, :, 0:4]
        X_cont = X[:, :, 4:9]
    else:
        # [0:3] -> [country_enc, region_enc, subregion_enc]
        # [3:8] -> [year_norm, life_exp_women, life_exp_men, life_exp_women_diff, life_exp_men_diff]
        X_cat = X[:, :, 0:3]
        X_cont = X[:, :, 3:8]

    # Further split categorical features for embeddings
    X_country = X_cat[:, :, 0]
    X_region  = X_cat[:, :, 1]
    X_subregion = X_cat[:, :, 2]
    if include_inter:
        X_inter = X_cat[:, :, 3]

    # Train-test split (time-based, no shuffle)
    if include_inter:
        X_country_train, X_country_test, y_train, y_test = train_test_split(X_country, y, test_size=0.2, shuffle=False)
        X_region_train,  X_region_test  = train_test_split(X_region,  test_size=0.2, shuffle=False)
        X_subregion_train, X_subregion_test = train_test_split(X_subregion, test_size=0.2, shuffle=False)
        X_inter_train, X_inter_test     = train_test_split(X_inter,   test_size=0.2, shuffle=False)
        X_cont_train,  X_cont_test      = train_test_split(X_cont,    test_size=0.2, shuffle=False)
    else:
        X_country_train, X_country_test, y_train, y_test = train_test_split(X_country, y, test_size=0.2, shuffle=False)
        X_region_train,  X_region_test  = train_test_split(X_region,  test_size=0.2, shuffle=False)
        X_subregion_train, X_subregion_test = train_test_split(X_subregion, test_size=0.2, shuffle=False)
        X_cont_train,  X_cont_test      = train_test_split(X_cont,    test_size=0.2, shuffle=False)

    # Determine vocabulary sizes
    num_country   = len(le_country.classes_)
    num_region    = len(le_region.classes_)
    num_subregion = len(le_subregion.classes_)
    if include_inter:
        num_inter = len(le_inter.classes_)
    
    # Build the model
    time_steps = X_cont_train.shape[1]  # should be same as the time_step param
    if include_inter:
        model = build_model(time_steps, num_country, num_region, num_subregion,
                            num_inter=num_inter, include_inter_region=True)
    else:
        model = build_model(time_steps, num_country, num_region, num_subregion,
                            include_inter_region=False)
    model.summary()

    # Prepare target values
    y_train_women = y_train[:, 0].reshape(-1, 1)
    y_train_men   = y_train[:, 1].reshape(-1, 1)
    y_test_women  = y_test[:, 0].reshape(-1, 1)
    y_test_men    = y_test[:, 1].reshape(-1, 1)

    # Setup callbacks and train the model

    # 原来只设置了一个回调，现在新增两个回调分别监控女性和男性输出的验证损失
    checkpoint_path_women = f'./GRU_predict/best_model_women_{experiment_type}.keras'
    checkpoint_path_men   = f'./GRU_predict/best_model_men_{experiment_type}.keras'
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    checkpoint_women = ModelCheckpoint(checkpoint_path_women, monitor='output_women_loss', save_best_only=True, verbose=1)
    checkpoint_men   = ModelCheckpoint(checkpoint_path_men, monitor='output_men_loss', save_best_only=True, verbose=1)
    
    if include_inter:
        history = model.fit(
            [X_country_train, X_region_train, X_subregion_train, X_inter_train, X_cont_train],
            [y_train_women, y_train_men],
            epochs=100, batch_size=32,
            callbacks=[early_stop, checkpoint_women, checkpoint_men]
        )
    else:
        history = model.fit(
            [X_country_train, X_region_train, X_subregion_train, X_cont_train],
            [y_train_women, y_train_men],
            epochs=100, batch_size=32,
            callbacks=[early_stop, checkpoint_women, checkpoint_men]
        )


    # Load the best models for women and men
    best_model_women = load_model(checkpoint_path_women, custom_objects={'reduce_sum_axis1': reduce_sum_axis1})
    best_model_men = load_model(checkpoint_path_men, custom_objects={'reduce_sum_axis1': reduce_sum_axis1})
    best_model_women.summary()
    best_model_men.summary()

    # --------------------- Prediction & Evaluation ---------------------
    # (A) With difference features
    if include_inter:
        pred_women_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test])[0]
        pred_men_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test])[1]
    else:
        pred_women_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test])[0]
        pred_men_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test])[1]
    predicted_values_diff = np.hstack((pred_women_diff, pred_men_diff))

    # (B) Without difference features: set diff columns in continuous input to 0
    X_cont_test_no_diff = np.copy(X_cont_test)
    # 最后两列是 diff
    X_cont_test_no_diff[:, :, 3:5] = 0

    if include_inter:
        pred_women_no_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test_no_diff])[0]
        pred_men_no_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test_no_diff])[1]
    else:
        pred_women_no_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test_no_diff])[0]
        pred_men_no_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test_no_diff])[1]
    predicted_values_no_diff = np.hstack((pred_women_no_diff, pred_men_no_diff))

    # Inverse transform predictions and true values (only for life expectancy columns)
    y_test_inv = scaler.inverse_transform(y_test)
    predicted_values_diff_inv = scaler.inverse_transform(predicted_values_diff)
    predicted_values_no_diff_inv = scaler.inverse_transform(predicted_values_no_diff)

    # Compute metrics
    _, mae_women_no_diff, r2_women_no_diff = compute_metrics(y_test_inv[:, 0], predicted_values_no_diff_inv[:, 0])
    _, mae_men_no_diff, r2_men_no_diff     = compute_metrics(y_test_inv[:, 1], predicted_values_no_diff_inv[:, 1])
    _, mae_women_diff, r2_women_diff       = compute_metrics(y_test_inv[:, 0], predicted_values_diff_inv[:, 0])
    _, mae_men_diff, r2_men_diff           = compute_metrics(y_test_inv[:, 1], predicted_values_diff_inv[:, 1])

    metrics_result = {
        'without_diff': {
            'MAE Women': mae_women_no_diff,
            'R2 Women':  r2_women_no_diff,
            'MAE Men':   mae_men_no_diff,
            'R2 Men':    r2_men_no_diff
        },
        'with_diff': {
            'MAE Women': mae_women_diff,
            'R2 Women':  r2_women_diff,
            'MAE Men':   mae_men_diff,
            'R2 Men':    r2_men_diff
        }
    }

    # **同时返回训练的历史（history.history），以便主程序画图**
    return metrics_result, history.history


if __name__ == "__main__":
    time_step = 5
    experiment_types = ['baseline', 'inter_unknown', 'inter_subregion']
    results = {}
    histories = {}

    for exp_type in experiment_types:
        metrics, hist = run_experiment(exp_type, time_step=time_step)
        results[exp_type] = metrics
        histories[exp_type] = hist

    # ======= 输出对比表 =======
    rows = []
    index_labels = []

    for exp_type, metrics in results.items():
        if exp_type == 'baseline':
            label = 'Baseline'
        elif exp_type == 'inter_unknown':
            label = 'Inter-region (unknown)'
        elif exp_type == 'inter_subregion':
            label = 'Inter-region (sub-region)'

        rows.append(metrics['without_diff'])
        index_labels.append(f"{label} - Without Diff")

        rows.append(metrics['with_diff'])
        index_labels.append(f"{label} - With Diff")

    comp_table = pd.DataFrame(rows, index=index_labels)
    print("\n========== Comparison of Evaluation Metrics (Original Scale) ==========")
    print(comp_table)
    
    comp_table.to_csv('experiment_results_Transformer.csv')