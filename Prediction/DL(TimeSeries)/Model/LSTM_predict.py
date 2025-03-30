import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, TimeDistributed
from tensorflow.keras.layers import Activation, Flatten, RepeatVector, Permute, Multiply, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def reduce_sum_axis1(x):
    return tf.reduce_sum(x, axis=1)


# Set random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 1. Load dataset
# 读取原始数据，并创建三个变体
# 变体1：不使用 inter-region 特征
# 变体2：使用 inter-region 特征，缺失值填充为 'Unknown'
# 变体3：使用 inter-region 特征，缺失值填充为对应的 sub-region

data = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')

# Variant 1: No Inter-region
data_no_inter = data[['country_name', 'region', 'sub-region', 'year', 'life_expectancy_women', 'life_expectancy_men']].copy()

# Variant 2: Inter-region with Unknown fill
data_inter_unknown = data[['country_name', 'region', 'sub-region', 'intermediate-region', 'year', 'life_expectancy_women', 'life_expectancy_men']].copy()
data_inter_unknown['intermediate-region'] = data_inter_unknown['intermediate-region'].fillna('Unknown')

# Variant 3: Inter-region with Sub-region fill
data_inter_sub = data[['country_name', 'region', 'sub-region', 'intermediate-region', 'year', 'life_expectancy_women', 'life_expectancy_men']].copy()
# 使用 sub-region 填充缺失的 intermediate-region
data_inter_sub['intermediate-region'] = data_inter_sub['intermediate-region'].fillna(data_inter_sub['sub-region'])


# 定义数据预处理函数
# 如果 with_inter 为 True，则会对 inter-region 进行 label encoding

def preprocess_data(df, with_inter=False):
    df['life_expectancy_women'] = pd.to_numeric(df['life_expectancy_women'].astype(str).str.replace(',', '.'), errors='coerce')
    df['life_expectancy_men'] = pd.to_numeric(df['life_expectancy_men'].astype(str).str.replace(',', '.'), errors='coerce')
    df['year'] = df['year'].astype(int)
    df['year_norm'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[['year']])
    df.sort_values(by=['country_name', 'year'], inplace=True)
    
    # Label encode categorical features
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_subregion = LabelEncoder()
    df['country_enc'] = le_country.fit_transform(df['country_name'].astype(str))
    df['region_enc'] = le_region.fit_transform(df['region'].astype(str))
    df['subregion_enc'] = le_subregion.fit_transform(df['sub-region'].astype(str))
    if with_inter:
        le_inter = LabelEncoder()
        df['inter_region_enc'] = le_inter.fit_transform(df['intermediate-region'].astype(str))
    
    # Normalize life expectancy values
    scaler_local = MinMaxScaler(feature_range=(0, 1))
    df[['life_expectancy_women', 'life_expectancy_men']] = scaler_local.fit_transform(df[['life_expectancy_women', 'life_expectancy_men']])
    
    # Compute first order differences
    df['life_expectancy_women_diff'] = df.groupby('country_name')['life_expectancy_women'].diff()
    df['life_expectancy_men_diff'] = df.groupby('country_name')['life_expectancy_men'].diff()
    df = df.dropna(subset=['life_expectancy_women_diff', 'life_expectancy_men_diff'])
    return df, scaler_local


# 修改 create_dataset 函数，增加参数 with_inter
# 如果 with_inter 为 True，则数据中包含 inter_region_enc 特征

def create_dataset(data, time_step=1, with_inter=False):
    X, y = [], []
    countries = data['country_name'].unique()
    for country in countries:
        country_data = data[data['country_name'] == country].sort_values(by='year')
        if with_inter:
            feat = country_data[['country_enc', 'region_enc', 'subregion_enc', 'inter_region_enc', 'year_norm',
                                  'life_expectancy_women', 'life_expectancy_men',
                                  'life_expectancy_women_diff', 'life_expectancy_men_diff']].values
        else:
            feat = country_data[['country_enc', 'region_enc', 'subregion_enc', 'year_norm',
                                  'life_expectancy_women', 'life_expectancy_men',
                                  'life_expectancy_women_diff', 'life_expectancy_men_diff']].values
        for i in range(len(feat) - time_step):
            X.append(feat[i:(i + time_step)])
            if with_inter:
                # 对于有 inter-region 的情况，目标值在索引 5 和 6
                target = feat[i + time_step, 5:7]
            else:
                target = feat[i + time_step, 4:6]
            y.append(target)
    return np.array(X), np.array(y)


# 构建模型函数，根据 with_inter 参数决定是否添加 inter-region 输入

def build_model(time_steps, num_country, num_region, num_subregion, num_inter=None, with_inter=False):
    input_country = Input(shape=(time_steps,), name="country_input")
    input_region = Input(shape=(time_steps,), name="region_input")
    input_subregion = Input(shape=(time_steps,), name="subregion_input")
    if with_inter:
        input_inter = Input(shape=(time_steps,), name="inter_region_input")
    input_cont = Input(shape=(time_steps, 5), name="continuous_input")  # 连续特征：year_norm, life_expectancy, life_expectancy, diff1, diff2
    
    emb_country = Embedding(input_dim=num_country, output_dim=16, input_length=time_steps)(input_country)
    emb_region = Embedding(input_dim=num_region, output_dim=8, input_length=time_steps)(input_region)
    emb_subregion = Embedding(input_dim=num_subregion, output_dim=8, input_length=time_steps)(input_subregion)
    
    if with_inter:
        emb_inter = Embedding(input_dim=num_inter, output_dim=8, input_length=time_steps)(input_inter)
        merged_cat = Concatenate(axis=-1)([emb_country, emb_region, emb_subregion, emb_inter])
    else:
        merged_cat = Concatenate(axis=-1)([emb_country, emb_region, emb_subregion])
        
    proj_cont = TimeDistributed(Dense(8, activation='relu'))(input_cont)
    
    merged = Concatenate(axis=-1)([merged_cat, proj_cont])
    
    x = LSTM(units=128, return_sequences=True)(merged)
    x = Dropout(0.3)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    # Attention 机制
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
    x = Multiply()([x, attention])
    
    x = Lambda(reduce_sum_axis1, output_shape=(64,))(x)
    
    x = Dense(units=32, activation='relu')(x)
    output_women = Dense(units=1, activation='linear', name='output_women')(x)
    output_men = Dense(units=1, activation='linear', name='output_men')(x)
    
    if with_inter:
        model = Model(inputs=[input_country, input_region, input_subregion, input_inter, input_cont], outputs=[output_women, output_men])
    else:
        model = Model(inputs=[input_country, input_region, input_subregion, input_cont], outputs=[output_women, output_men])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


# 定义实验运行函数
# 对给定的预处理数据，根据 time_step 构建时序数据，并划分训练/测试集，训练模型后评估两种情况：
# 1. 使用包含差分特征的测试数据（With Difference）
# 2. 对测试数据将差分特征置零（Without Difference）
# 返回 MAE 和 R2 指标

def run_experiment(df, scaler_local, variant_name, with_inter=False, time_step=5):
    X, y = create_dataset(df, time_step=time_step, with_inter=with_inter)
    
    # 根据是否包含 inter-region 调整特征切分
    if with_inter:
        # X shape: (samples, time_step, 9)
        X_cat = X[:, :, 0:4]
        X_cont = X[:, :, 4:9]
        X_country = X_cat[:, :, 0]
        X_region = X_cat[:, :, 1]
        X_subregion = X_cat[:, :, 2]
        X_inter = X_cat[:, :, 3]
    else:
        # X shape: (samples, time_step, 8)
        X_cat = X[:, :, 0:3]
        X_cont = X[:, :, 3:8]
        X_country = X_cat[:, :, 0]
        X_region = X_cat[:, :, 1]
        X_subregion = X_cat[:, :, 2]
    
    # 保持时间顺序划分训练/测试集，80% 训练，20% 测试
    split_idx = int(0.8 * X_country.shape[0])
    X_country_train, X_country_test = X_country[:split_idx], X_country[split_idx:]
    X_region_train, X_region_test = X_region[:split_idx], X_region[split_idx:]
    X_subregion_train, X_subregion_test = X_subregion[:split_idx], X_subregion[split_idx:]
    if with_inter:
        X_inter_train, X_inter_test = X_inter[:split_idx], X_inter[split_idx:]
    X_cont_train, X_cont_test = X_cont[:split_idx], X_cont[split_idx:]
    
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # 确定各类别的词汇表大小
    num_country = df['country_enc'].nunique()
    num_region = df['region_enc'].nunique()
    num_subregion = df['subregion_enc'].nunique()
    if with_inter:
        num_inter = df['inter_region_enc'].nunique()
    
    time_steps = X_country_train.shape[1]
    if with_inter:
        model = build_model(time_steps, num_country, num_region, num_subregion, num_inter, with_inter=True)
    else:
        model = build_model(time_steps, num_country, num_region, num_subregion, with_inter=False)
    
    # 修改模型检查点
    checkpoint_file_women = f'./LSTM_predict/best_model_lstm_women_{variant_name.replace(" ", "_")}.keras'
    checkpoint_file_men   = f'./LSTM_predict/best_model_lstm_men_{variant_name.replace(" ", "_")}.keras'
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_women = ModelCheckpoint(checkpoint_file_women, monitor='val_output_women_loss', save_best_only=True, verbose=1)
    checkpoint_men   = ModelCheckpoint(checkpoint_file_men, monitor='val_output_men_loss', save_best_only=True, verbose=1)
    
    # 训练模型
    if with_inter:
        history = model.fit(
            [X_country_train, X_region_train, X_subregion_train, X_inter_train, X_cont_train],
            [y_train[:, 0].reshape(-1, 1), y_train[:, 1].reshape(-1, 1)],
            epochs=100, batch_size=32,
            validation_data=(
                [X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test],
                [y_test[:, 0].reshape(-1, 1), y_test[:, 1].reshape(-1, 1)]
            ),
            callbacks=[early_stop, checkpoint_women, checkpoint_men],
            verbose=1
        )
    else:
        history = model.fit(
            [X_country_train, X_region_train, X_subregion_train, X_cont_train],
            [y_train[:, 0].reshape(-1, 1), y_train[:, 1].reshape(-1, 1)],
            epochs=100, batch_size=32,
            validation_data=(
                [X_country_test, X_region_test, X_subregion_test, X_cont_test],
                [y_test[:, 0].reshape(-1, 1), y_test[:, 1].reshape(-1, 1)]
            ),
            callbacks=[early_stop, checkpoint_women, checkpoint_men],
            verbose=1
        )
    
    # 加载最优模型，分别加载女性和男性的最佳模型
    best_model_women = load_model(checkpoint_file_women)
    best_model_men   = load_model(checkpoint_file_men)
    
    # 预测：两种情况
    # (1) 使用包含差分特征的测试数据
    if with_inter:
        # 分别使用两个模型进行预测
        pred_women_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test], verbose=0)[0]
        pred_men_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test], verbose=0)[1]
    else:
        pred_women_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test], verbose=0)[0]
        pred_men_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test], verbose=0)[1]
    pred_diff = np.hstack((pred_women_diff, pred_men_diff))
    
    # (2) 不使用差分特征：将测试数据中的差分列置为 0（最后两列）
    X_cont_test_no_diff = np.copy(X_cont_test)
    X_cont_test_no_diff[:, :, -2:] = 0
    if with_inter:
        pred_women_no_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test_no_diff], verbose=0)[0]
        pred_men_no_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_inter_test, X_cont_test_no_diff], verbose=0)[1]
    else:
        pred_women_no_diff = best_model_women.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test_no_diff], verbose=0)[0]
        pred_men_no_diff   = best_model_men.predict([X_country_test, X_region_test, X_subregion_test, X_cont_test_no_diff], verbose=0)[1]
    pred_no_diff = np.hstack((pred_women_no_diff, pred_men_no_diff))
    
    # 使用 scaler_local 逆转换得到原始尺度
    y_test_inv = scaler_local.inverse_transform(y_test)
    pred_diff_inv = scaler_local.inverse_transform(pred_diff)
    pred_no_diff_inv = scaler_local.inverse_transform(pred_no_diff)
    
    # 计算 MAE 和 R2 指标
    results = {}
    # Without Difference
    mae_women_no = mean_absolute_error(y_test_inv[:, 0], pred_no_diff_inv[:, 0])
    r2_women_no = r2_score(y_test_inv[:, 0], pred_no_diff_inv[:, 0])
    mae_men_no = mean_absolute_error(y_test_inv[:, 1], pred_no_diff_inv[:, 1])
    r2_men_no = r2_score(y_test_inv[:, 1], pred_no_diff_inv[:, 1])
    results["Without Difference"] = [mae_women_no, r2_women_no, mae_men_no, r2_men_no]

    # With Difference
    mae_women_diff = mean_absolute_error(y_test_inv[:, 0], pred_diff_inv[:, 0])
    r2_women_diff = r2_score(y_test_inv[:, 0], pred_diff_inv[:, 0])
    mae_men_diff = mean_absolute_error(y_test_inv[:, 1], pred_diff_inv[:, 1])
    r2_men_diff = r2_score(y_test_inv[:, 1], pred_diff_inv[:, 1])
    results["With Difference"] = [mae_women_diff, r2_women_diff, mae_men_diff, r2_men_diff]

    # 返回评估指标和训练的 history
    return results, history.history


# 运行所有实验，并对比结果
experiment_results = []
history_dict = {}  # 定义 history_dict 用于保存每个实验的训练历史

experiments = [
    {"name": "No Inter-region", "df": data_no_inter, "with_inter": False},
    {"name": "Inter-region (Unknown)", "df": data_inter_unknown, "with_inter": True},
    {"name": "Inter-region (Sub-region)", "df": data_inter_sub, "with_inter": True}
]

for exp in experiments:
    df_processed, scaler_local = preprocess_data(exp["df"], with_inter=exp["with_inter"])
    # 假设 run_experiment 返回两个值：results 和 history
    results, hist = run_experiment(df_processed, scaler_local, exp["name"], with_inter=exp["with_inter"], time_step=5)
    history_dict[exp["name"]] = hist  # 保存训练 history
    for setting, metrics in results.items():
        experiment_results.append({
            "Experiment": exp["name"] + " - " + setting,
            "MAE Women": metrics[0],
            "R2 Women": metrics[1],
            "MAE Men": metrics[2],
            "R2 Men": metrics[3]
        })

# 构造评估对比表格，并打印
eval_table = pd.DataFrame(experiment_results)
print("\n========== LSTM Evaluation Metrics Comparison (MAE & R2) ==========")
print(eval_table)

# 保存结果为 CSV 文件
eval_table.to_csv("experiment_results_LSTM.csv", index=False)


