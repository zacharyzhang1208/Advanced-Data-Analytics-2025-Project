import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def reduce_sum_axis1(x):
    return tf.reduce_sum(x, axis=1)

def preprocess_data(df):
    # Convert life expectancy columns to numeric (handling decimal commas)
    df['life_expectancy_women'] = pd.to_numeric(df['life_expectancy_women'].astype(str).str.replace(',', '.'), errors='coerce')
    df['life_expectancy_men']   = pd.to_numeric(df['life_expectancy_men'].astype(str).str.replace(',', '.'), errors='coerce')
    df['year'] = df['year'].astype(int)
    df['year_norm'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[['year']])
    df.sort_values(by=['country_name', 'year'], inplace=True)
    
    # Label encode categorical features
    le_country = LabelEncoder()
    le_region  = LabelEncoder()
    le_sub    = LabelEncoder()
    df['country_enc'] = le_country.fit_transform(df['country_name'].astype(str))
    df['region_enc']  = le_region.fit_transform(df['region'].astype(str))
    df['subregion_enc'] = le_sub.fit_transform(df['sub-region'].astype(str))
    
    # Normalize life expectancy values
    scaler_local = MinMaxScaler(feature_range=(0, 1))
    df[['life_expectancy_women', 'life_expectancy_men']] = scaler_local.fit_transform(
        df[['life_expectancy_women', 'life_expectancy_men']]
    )
    
    # Compute first-order differences
    df['life_expectancy_women_diff'] = df.groupby('country_name')['life_expectancy_women'].diff()
    df['life_expectancy_men_diff']   = df.groupby('country_name')['life_expectancy_men'].diff()
    df = df.dropna(subset=['life_expectancy_women_diff', 'life_expectancy_men_diff'])
    return df, scaler_local

def create_dataset_with_meta(data, time_step=5):
    X, y, meta = [], [], []
    countries = data['country_name'].unique()
    for country in countries:
        country_data = data[data['country_name'] == country].sort_values(by='year')
        # Features: country_enc, region_enc, subregion_enc, year_norm,
        #           life_expectancy_women, life_expectancy_men, diff_women, diff_men
        feat = country_data[['country_enc', 'region_enc', 'subregion_enc', 'year_norm',
                             'life_expectancy_women', 'life_expectancy_men',
                             'life_expectancy_women_diff', 'life_expectancy_men_diff']].values
        # Meta information for visualization: country_name, region, sub-region, year
        meta_data = country_data[['country_name', 'region', 'sub-region', 'year']].reset_index(drop=True)
        true_vals = country_data[['life_expectancy_women', 'life_expectancy_men']].reset_index(drop=True)
        for i in range(len(feat) - time_step):
            X.append(feat[i:(i + time_step)])
            # Target values: women and men life expectancy at row i+time_step
            y.append(feat[i + time_step, 4:6])
            meta_row = meta_data.iloc[i + time_step].to_dict()
            meta_row['true_women_scaled'] = true_vals.iloc[i + time_step, 0]
            meta_row['true_men_scaled']   = true_vals.iloc[i + time_step, 1]
            meta.append(meta_row)
    return np.array(X), np.array(y), meta

def main():
    # 1. Load dataset and select No Inter-region variant
    df = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')
    df_no_inter = df[['country_name', 'region', 'sub-region', 'year', 'life_expectancy_women', 'life_expectancy_men']].copy()
    
    # 2. Preprocess data
    df_processed, scaler_local = preprocess_data(df_no_inter)
    
    # 3. Create dataset with meta information (time_step = 5)
    time_step = 5
    X, y, meta = create_dataset_with_meta(df_processed, time_step=time_step)
    
    # 4. Split into test set (using 80%/20% split)
    split_idx = int(0.8 * len(X))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    meta_test = meta[split_idx:]
    
    # 5. Prepare model inputs for No Inter-region variant:
    #    Inputs: [country, region, subregion, continuous features]
    X_cat_test = X_test[:, :, 0:3]  # country_enc, region_enc, subregion_enc
    X_cont_test = X_test[:, :, 3:8]  # year_norm, life_expectancy_women, life_expectancy_men, diff_women, diff_men
    X_country_test = X_cat_test[:, :, 0]
    X_region_test  = X_cat_test[:, :, 1]
    X_subregion_test = X_cat_test[:, :, 2]
    
    # 6. Load best models (differential + no inter-region)
    best_model_women = load_model('./LSTM_predict/best_model_lstm_women_No_Inter-region.keras')
    best_model_men   = load_model('./LSTM_predict/best_model_lstm_men_No_Inter-region.keras')
    
    # 7. Get predictions from models
    pred_women = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test], verbose=0)[0]
    pred_men   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test], verbose=0)[1]
    
    # 8. Inverse transform predictions and true values
    y_test_inv = scaler_local.inverse_transform(y_test)
    preds = np.hstack((pred_women, pred_men))
    preds_inv = scaler_local.inverse_transform(preds)
    
    # 9. Build meta DataFrame for visualization
    meta_df = pd.DataFrame(meta_test)
    meta_df['year'] = meta_df['year'].astype(int)
    meta_df['true_women'] = y_test_inv[:, 0]
    meta_df['pred_women'] = preds_inv[:, 0]
    meta_df['true_men']   = y_test_inv[:, 1]
    meta_df['pred_men']   = preds_inv[:, 1]
    
    # Visualization layout: Top row with 3 subplots, bottom row with 2 subplots (centered)
    # Define regions based on unique continent values from meta_df (expected 5 regions)
    regions = sorted(meta_df['region'].unique())
    if len(regions) != 5:
        print(f"Warning: Expected 5 regions but found {len(regions)}. Layout may not be as intended.")
    top_regions = regions[:3]
    bottom_regions = regions[3:]
    
    # --- Women's Predictions ---
    fig_women = plt.figure(figsize=(15,8))
    # Define positions (in figure coordinates) for the top row (3 subplots)
    # (left, bottom, width, height)
    top_positions = [(0.05, 0.55, 0.27, 0.35), (0.37, 0.55, 0.27, 0.35), (0.69, 0.55, 0.27, 0.35)]
    # Define positions for the bottom row (2 subplots), centered relative to the top row
    bottom_positions = [(0.195, 0.1, 0.27, 0.35), (0.535, 0.1, 0.27, 0.35)]
    
    # Plot top row for Women's predictions
    for pos, region in zip(top_positions, top_regions):
        ax = fig_women.add_axes(pos)
        region_data = meta_df[meta_df['region'] == region]
        # Aggregate by year to get mean true and predicted values
        region_group = region_data.groupby('year').agg({'true_women': 'mean', 'pred_women': 'mean'}).reset_index()
        ax.plot(region_group['year'], region_group['true_women'], label='True')
        ax.plot(region_group['year'], region_group['pred_women'], label='Predicted')
        ax.set_xlabel('Year')
        ax.set_ylabel('Life Expectancy (Women)')
        ax.set_title(region)
        ax.legend()
        ax.grid(True)
    
    # Plot bottom row for Women's predictions
    for pos, region in zip(bottom_positions, bottom_regions):
        ax = fig_women.add_axes(pos)
        region_data = meta_df[meta_df['region'] == region]
        region_group = region_data.groupby('year').agg({'true_women': 'mean', 'pred_women': 'mean'}).reset_index()
        ax.plot(region_group['year'], region_group['true_women'], label='True')
        ax.plot(region_group['year'], region_group['pred_women'], label='Predicted')
        ax.set_xlabel('Year')
        ax.set_ylabel('Life Expectancy (Women)')
        ax.set_title(region)
        ax.legend()
        ax.grid(True)
    
    fig_women.suptitle("Women's Life Expectancy Predictions by Continent", fontsize=16)
    plt.show()
    
    # --- Men's Predictions ---
    fig_men = plt.figure(figsize=(15,8))
    # Use the same layout positions as for Women's predictions
    # Top row positions for men's predictions
    for pos, region in zip(top_positions, top_regions):
        ax = fig_men.add_axes(pos)
        region_data = meta_df[meta_df['region'] == region]
        region_group = region_data.groupby('year').agg({'true_men': 'mean', 'pred_men': 'mean'}).reset_index()
        ax.plot(region_group['year'], region_group['true_men'], label='True')
        ax.plot(region_group['year'], region_group['pred_men'], label='Predicted')
        ax.set_xlabel('Year')
        ax.set_ylabel('Life Expectancy (Men)')
        ax.set_title(region)
        ax.legend()
        ax.grid(True)
    
    # Bottom row positions for men's predictions
    for pos, region in zip(bottom_positions, bottom_regions):
        ax = fig_men.add_axes(pos)
        region_data = meta_df[meta_df['region'] == region]
        region_group = region_data.groupby('year').agg({'true_men': 'mean', 'pred_men': 'mean'}).reset_index()
        ax.plot(region_group['year'], region_group['true_men'], label='True')
        ax.plot(region_group['year'], region_group['pred_men'], label='Predicted')
        ax.set_xlabel('Year')
        ax.set_ylabel('Life Expectancy (Men)')
        ax.set_title(region)
        ax.legend()
        ax.grid(True)
    
    fig_men.suptitle("Men's Life Expectancy Predictions by Continent", fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()