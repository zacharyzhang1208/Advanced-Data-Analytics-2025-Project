import random
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, LayerNormalization, GlobalAveragePooling1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ===================== 1. 数据加载与预处理 =====================
# 仅保留输入特征和目标： 'country_name', 'region', 'sub-region', 'intermediate-region', 'year', 'life_expectancy_women'
data = pd.read_csv('life_expectancy_dataset.csv', sep=';', decimal=',')
data = data[['country_name', 'region', 'sub-region', 'intermediate-region', 'year', 'life_expectancy_women', 'life_expectancy_men']]

# 转换目标为数值
data['life_expectancy_women'] = pd.to_numeric(data['life_expectancy_women'].astype(str).str.replace(',', '.'), errors='coerce')
data['life_expectancy_men'] = pd.to_numeric(data['life_expectancy_men'].astype(str).str.replace(',', '.'), errors='coerce')
data['year'] = data['year'].astype(int)
data.sort_values(by=['country_name', 'year'], inplace=True)

# 编码类别特征
le_country = LabelEncoder()
le_region = LabelEncoder()
le_subregion = LabelEncoder()
data['country_enc'] = le_country.fit_transform(data['country_name'].astype(str))
data['region_enc'] = le_region.fit_transform(data['region'].astype(str))
data['subregion_enc'] = le_subregion.fit_transform(data['sub-region'].astype(str))

# 对 intermediate-region 进行编码
le_intermediate = LabelEncoder()
data['intermediate_enc'] = le_intermediate.fit_transform(data['intermediate-region'].astype(str))

# 对年份进行归一化（映射到[0,1]）
min_year = data['year'].min()
max_year = data['year'].max()
data['year_norm'] = (data['year'] - min_year) / (max_year - min_year)
# 对预期寿命进行归一化处理（映射到[0,1]），生成新的列 'life_expectancy_women_norm'
target_scaler = MinMaxScaler(feature_range=(0, 1))
data['life_expectancy_women_norm'] = target_scaler.fit_transform(data[['life_expectancy_women']])
target_scaler_men = MinMaxScaler(feature_range=(0, 1))
data['life_expectancy_men_norm'] = target_scaler_men.fit_transform(data[['life_expectancy_men']])

# ===================== 2. 构造序列数据 =====================
def create_sequences(data, time_step=5, target_col='life_expectancy_women_norm', include_inter=False, differencing=False):
    X, y, years = [], [], []
    base_vals = []  # 用于存储差分模式下的基础值
    # 根据是否使用 intermediate 特征选择列
    feature_cols = ['country_enc', 'region_enc', 'subregion_enc', 'year_norm']
    if include_inter:
        # 插入 intermediate 编码，在 subregion 和 year_norm 之间
        feature_cols.insert(3, 'intermediate_enc')

    countries = data['country_name'].unique()
    for country in countries:
        country_data = data[data['country_name'] == country].sort_values(by='year')
        if len(country_data) < time_step + 1:
            continue
        features = country_data[feature_cols].values
        target = country_data[target_col].values
        year_seq = country_data['year'].values
        for i in range(len(features) - time_step):
            X.append(features[i:i+time_step])
            years.append(year_seq[i+time_step])
            if differencing:
                # 目标为相邻时刻的差分
                y.append(target[i+time_step] - target[i+time_step-1])
                base_vals.append(target[i+time_step-1])
            else:
                y.append(target[i+time_step])
    if differencing:
        return np.array(X), np.array(y), np.array(base_vals), np.array(years)
    else:
        return np.array(X), np.array(y), np.array(years)

time_step = 5
X_all_women, y_all_women, years_all = create_sequences(data, time_step, target_col='life_expectancy_women_norm')
X_all_men, y_all_men, _ = create_sequences(data, time_step, target_col='life_expectancy_men_norm')

# 划分数据：前53年训练，后10年测试
cutoff_year = min_year + 53 - 1  # 例如，如果min_year=1960, cutoff_year=2012
print("Cutoff year:", cutoff_year)

train_idx = np.where(years_all <= cutoff_year)[0]
test_idx = np.where(years_all > cutoff_year)[0]

X_train_women, y_train_women = X_all_women[train_idx], y_all_women[train_idx]
X_test_women, y_test_women = X_all_women[test_idx], y_all_women[test_idx]

X_train_men, y_train_men = X_all_men[train_idx], y_all_men[train_idx]
X_test_men, y_test_men = X_all_men[test_idx], y_all_men[test_idx]

print("X_train_women shape:", X_train_women.shape)
print("X_test_women shape:", X_test_women.shape)
print("X_train_men shape:", X_train_men.shape)
print("y_test_men shape:", y_test_men.shape)

# ===================== 3. Transformer 模型构建 =====================
# 定义 Transformer 相关组件

# Transformer Block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Positional Encoding
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, embed_dim)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# 定义一个层对输入的4个特征进行嵌入
class FeatureEmbedding(Layer):
    def __init__(self, num_countries, num_regions, num_subregions, num_intermediates=None, embed_dims=(16,8,8), year_embed_dim=4, inter_embed_dim=8, include_inter=False, **kwargs):
        super(FeatureEmbedding, self).__init__(**kwargs)
        self.num_countries = num_countries
        self.num_regions = num_regions
        self.num_subregions = num_subregions
        self.num_intermediates = num_intermediates
        self.embed_dims = embed_dims
        self.year_embed_dim = year_embed_dim
        self.inter_embed_dim = inter_embed_dim
        self.include_inter = include_inter

        self.country_embedding = Embedding(input_dim=num_countries, output_dim=embed_dims[0])
        self.region_embedding = Embedding(input_dim=num_regions, output_dim=embed_dims[1])
        self.subregion_embedding = Embedding(input_dim=num_subregions, output_dim=embed_dims[2])
        if self.include_inter:
            self.intermediate_embedding = Embedding(input_dim=num_intermediates, output_dim=inter_embed_dim)
        self.year_dense = Dense(year_embed_dim, activation='linear')

    def get_config(self):
        config = super(FeatureEmbedding, self).get_config()
        config.update({
            'num_countries': self.num_countries,
            'num_regions': self.num_regions,
            'num_subregions': self.num_subregions,
            'num_intermediates': self.num_intermediates,
            'embed_dims': self.embed_dims,
            'year_embed_dim': self.year_embed_dim,
            'inter_embed_dim': self.inter_embed_dim,
            'include_inter': self.include_inter,
        })
        return config

    def call(self, inputs):
        if self.include_inter:
            # inputs shape: (batch, time_step, 5) -> [country, region, subregion, intermediate, year]
            country = inputs[:,:,0]
            region = inputs[:,:,1]
            subregion = inputs[:,:,2]
            intermediate = inputs[:,:,3]
            year = inputs[:,:,4]
            country_emb = self.country_embedding(tf.cast(country, tf.int32))
            region_emb = self.region_embedding(tf.cast(region, tf.int32))
            subregion_emb = self.subregion_embedding(tf.cast(subregion, tf.int32))
            intermediate_emb = self.intermediate_embedding(tf.cast(intermediate, tf.int32))
            year_emb = self.year_dense(tf.expand_dims(year, -1))
            return tf.concat([country_emb, region_emb, subregion_emb, intermediate_emb, year_emb], axis=-1)
        else:
            # inputs shape: (batch, time_step, 4)
            country = inputs[:,:,0]
            region = inputs[:,:,1]
            subregion = inputs[:,:,2]
            year = inputs[:,:,3]
            country_emb = self.country_embedding(tf.cast(country, tf.int32))
            region_emb = self.region_embedding(tf.cast(region, tf.int32))
            subregion_emb = self.subregion_embedding(tf.cast(subregion, tf.int32))
            year_emb = self.year_dense(tf.expand_dims(year, -1))
            return tf.concat([country_emb, region_emb, subregion_emb, year_emb], axis=-1)
        
def build_transformer_model(sequence_length, num_countries, num_regions, num_subregions,
                            num_intermediates=None,
                            embed_dims=(16,8,8), year_embed_dim=4, num_heads=4, ff_dim=128,
                            dropout_rate=0.1, include_inter=False, inter_embed_dim=8):
    if include_inter:
         inputs = Input(shape=(sequence_length, 5))
         x = FeatureEmbedding(num_countries, num_regions, num_subregions, num_intermediates=num_intermediates, embed_dims=embed_dims, year_embed_dim=year_embed_dim, inter_embed_dim=inter_embed_dim, include_inter=True)(inputs)
         embed_dim = embed_dims[0] + embed_dims[1] + embed_dims[2] + inter_embed_dim + year_embed_dim
    else:
         inputs = Input(shape=(sequence_length, 4))
         x = FeatureEmbedding(num_countries, num_regions, num_subregions, embed_dims=embed_dims, year_embed_dim=year_embed_dim, include_inter=False)(inputs)
         embed_dim = embed_dims[0] + embed_dims[1] + embed_dims[2] + year_embed_dim
    x = PositionalEncoding(sequence_length, embed_dim)(x)
    # Transformer Block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='linear')(x)  # 预测 life_expectancy_women
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

# 获取嵌入层需要的类别数
num_countries = data['country_name'].nunique()
num_regions = data['region'].nunique()
num_subregions = data['sub-region'].nunique()
num_intermediates = data['intermediate_enc'].nunique()

sequence_length = X_train_women.shape[1]  # time_step
model_women = build_transformer_model(sequence_length, num_countries, num_regions, num_subregions,
                                num_intermediates=num_intermediates,
                                embed_dims=(16,8,8), year_embed_dim=4, num_heads=4, ff_dim=128, dropout_rate=0.1)
model_men = build_transformer_model(sequence_length, num_countries, num_regions, num_subregions,
                                num_intermediates=num_intermediates,
                                embed_dims=(16,8,8), year_embed_dim=4, num_heads=4, ff_dim=128, dropout_rate=0.1)
model_women.summary()
model_men.summary()

## Model training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_women = ModelCheckpoint('./transformer_predict/best_transformer_model_women.keras',
                                   monitor='val_loss', save_best_only=True, verbose=1)
history_women = model_women.fit(X_train_women, y_train_women, epochs=100, batch_size=32,
                    validation_data=(X_test_women, y_test_women), callbacks=[early_stop, checkpoint_women])
checkpoint_men = ModelCheckpoint('./transformer_predict/best_transformer_model_men.keras',
                                 monitor='val_loss', save_best_only=True, verbose=1)
history_men = model_men.fit(X_train_men, y_train_men, epochs=100, batch_size=32,
                    validation_data=(X_test_men, y_test_men), callbacks=[early_stop, checkpoint_men])



# ===================== 4. 实验对比 =====================
experiments = [
    {"name": "baseline", "include_inter": False, "fill_method": None},
    {"name": "inter_unknown", "include_inter": True, "fill_method": "unknown"},
    {"name": "inter_subregion", "include_inter": True, "fill_method": "subregion"}
]
results = {}

# 新增两个列表，存储每次实验的训练历史
history_women_all = []
history_men_all = []

for exp in experiments:
    exp_type = exp["name"]
    include_inter = exp["include_inter"]
    fill_method = exp["fill_method"]
    
    metrics_result = {}
    for diff in [False, True]:
        # 准备实验数据
        data_exp = data.copy()
        if include_inter:
            if fill_method == "unknown":
                data_exp['intermediate-region'] = data_exp['intermediate-region'].fillna("Unknown")
            elif fill_method == "subregion":
                data_exp['intermediate-region'] = data_exp['intermediate-region'].fillna(data_exp['sub-region'])
            le_inter = LabelEncoder()
            data_exp['intermediate_enc'] = le_inter.fit_transform(data_exp['intermediate-region'].astype(str))
        
        # 创建序列
        if not diff:
            X_women_exp, y_women_exp, years_exp = create_sequences(data_exp, time_step, target_col='life_expectancy_women_norm', include_inter=include_inter, differencing=False)
            X_men_exp, y_men_exp, _ = create_sequences(data_exp, time_step, target_col='life_expectancy_men_norm', include_inter=include_inter, differencing=False)
        else:
            X_women_exp, y_women_exp, base_women, years_exp = create_sequences(data_exp, time_step, target_col='life_expectancy_women_norm', include_inter=include_inter, differencing=True)
            X_men_exp, y_men_exp, base_men, _ = create_sequences(data_exp, time_step, target_col='life_expectancy_men_norm', include_inter=include_inter, differencing=True)
        
        # 根据年份划分训练和测试集
        train_idx = np.where(years_exp <= cutoff_year)[0]
        test_idx = np.where(years_exp > cutoff_year)[0]
        
        X_train_women = X_women_exp[train_idx]
        y_train_women = y_women_exp[train_idx]
        X_test_women = X_women_exp[test_idx]
        y_test_women = y_women_exp[test_idx]
        if diff:
            base_test_women = base_women[test_idx]
        
        X_train_men = X_men_exp[train_idx]
        y_train_men = y_men_exp[train_idx]
        X_test_men = X_men_exp[test_idx]
        y_test_men = y_men_exp[test_idx]
        if diff:
            base_test_men = base_men[test_idx]
        
        # 获取类别数
        num_countries_exp = data_exp['country_name'].nunique()
        num_regions_exp = data_exp['region'].nunique()
        num_subregions_exp = data_exp['sub-region'].nunique()
        if include_inter:
            num_intermediates_exp = data_exp['intermediate_enc'].nunique()
        else:
            num_intermediates_exp = None
        
        # 构建模型
        model_women_exp = build_transformer_model(sequence_length, num_countries_exp, num_regions_exp, num_subregions_exp,
                                                  num_intermediates=num_intermediates_exp,
                                                  embed_dims=(16,8,8), year_embed_dim=4, num_heads=4, ff_dim=128, dropout_rate=0.1,
                                                  include_inter=include_inter, inter_embed_dim=8)
        model_men_exp = build_transformer_model(sequence_length, num_countries_exp, num_regions_exp, num_subregions_exp,
                                                num_intermediates=num_intermediates_exp,
                                                embed_dims=(16,8,8), year_embed_dim=4, num_heads=4, ff_dim=128, dropout_rate=0.1,
                                                include_inter=include_inter, inter_embed_dim=8)
        
        # 模型训练，保存训练历史
        # 根据实验类型和是否使用差分构造检查点文件名
        checkpoint_file_women_exp = f'./transformer_predict/best_transformer_model_women_{exp_type}_' \
                                      f'{"with_diff" if diff else "without_diff"}.keras'
        checkpoint_file_men_exp   = f'./transformer_predict/best_transformer_model_men_{exp_type}_' \
                                      f'{"with_diff" if diff else "without_diff"}.keras'
        early_stop_exp = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint_women_exp = ModelCheckpoint(checkpoint_file_women_exp, monitor='val_loss', save_best_only=True, verbose=0)
        history_women_exp = model_women_exp.fit(X_train_women, y_train_women, epochs=100, batch_size=32,
                            validation_data=(X_test_women, y_test_women), callbacks=[early_stop_exp, checkpoint_women_exp], verbose=0)
        checkpoint_men_exp = ModelCheckpoint(checkpoint_file_men_exp, monitor='val_loss', save_best_only=True, verbose=0)
        history_men_exp = model_men_exp.fit(X_train_men, y_train_men, epochs=100, batch_size=32,
                            validation_data=(X_test_men, y_test_men), callbacks=[early_stop_exp, checkpoint_men_exp], verbose=0)
        
        # 保存当前实验的训练历史
        experiment_label = f"{exp_type} - {'With Diff' if diff else 'Without Diff'}"
        history_women_all.append((experiment_label, history_women_exp.history))
        history_men_all.append((experiment_label, history_men_exp.history))
        
        # 预测
        y_pred_women = model_women_exp.predict(X_test_women).squeeze()
        y_pred_men = model_men_exp.predict(X_test_men).squeeze()
        if diff:
            y_pred_women = y_pred_women + base_test_women
            y_pred_men = y_pred_men + base_test_men
            y_test_women = y_test_women + base_test_women
            y_test_men = y_test_men + base_test_men
        
        # 计算指标：MAE 和 R²
        # 首先对预测值和真实值进行逆归一化还原到原始数据尺度
        y_test_women_orig = target_scaler.inverse_transform(y_test_women.reshape(-1, 1)).squeeze()
        y_pred_women_orig = target_scaler.inverse_transform(y_pred_women.reshape(-1, 1)).squeeze()
        y_test_men_orig = target_scaler_men.inverse_transform(y_test_men.reshape(-1, 1)).squeeze()
        y_pred_men_orig = target_scaler_men.inverse_transform(y_pred_men.reshape(-1, 1)).squeeze()

        mae_women = mean_absolute_error(y_test_women_orig, y_pred_women_orig)
        r2_women = r2_score(y_test_women_orig, y_pred_women_orig)
        mae_men = mean_absolute_error(y_test_men_orig, y_pred_men_orig)
        r2_men = r2_score(y_test_men_orig, y_pred_men_orig)
        metrics_dict = {
            'MAE Women': mae_women,
            'R2 Women': r2_women,
            'MAE Men': mae_men,
            'R2 Men': r2_men
        }
        if diff:            
            metrics_result["with_diff"] = metrics_dict
        else:
            metrics_result["without_diff"] = metrics_dict
    results[exp_type] = metrics_result

# 构建对比表格
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
print("\n========== Transformer Comparison of Evaluation Metrics (Original Scale) ==========")
print(comp_table)
comp_table.to_csv('experiment_results_Transformer.csv')