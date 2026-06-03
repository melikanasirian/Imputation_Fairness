#!/usr/bin/env python
# coding: utf-8
import json

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import sys
import torch

from utils import pick_epsilon, MAE, RMSE
from CMI_torch import compute_all_cmi_methods, estimate_CMI_soft_kronecker_gaussian, estimate_CMI_gumbel_softmax_kernel, \
    estimate_CMI_separate_kernel, generate_distributions_for_discrete_data
from Sinkhorn_CMI import SinkhornImputation_CMI
from SinkhornImputation import SinkhornImputation
from Inject_Missing_Values import Inject_Missing_Values
from sklearn.impute import SimpleImputer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
run_id = sys.argv[1] if len(sys.argv) >= 2 else 'experiment_results'
sample = sys.argv[2] if len(sys.argv) >= 3 else '128'

sample_size = None
sample_name = None
if sample.isnumeric():
    sample_size = int(sample)
else:
    sample_name = sample

iterations = int(sys.argv[3]) if len(sys.argv) >= 4 else 100

# options ['mean', 'mice', 'soft', 'sk', 'sk_cmi', 'sk_cmi0']
imp_methods = sys.argv[4:] if len(sys.argv) >= 5 else ['sk_cmi']

adult_data_train = pd.read_csv(os.path.join('Data', 'adult', 'adult.data'),
                               header=None,
                               delimiter=',')
adult_data_test = pd.read_csv(os.path.join('Data', 'adult', 'adult.test'),
                              header=None,
                              delimiter=',', skiprows=1)
adult_data = pd.concat([adult_data_train, adult_data_test], axis=0).reset_index(drop=True)

# Replace all ' ?' values with np.nan
adult_data = adult_data.replace(' ?', np.nan)

missing_percentage = (adult_data.isnull().sum() / len(adult_data)) * 100
print(missing_percentage)

adult_data = adult_data.dropna()

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
adult_data.columns = columns

adult_data.nunique()

## Fix income : [<=50K' ' >50K' ' <=50K.' ' >50K.] -----> [ ' >50K',  ' <=50K']

adult_data['income'] = adult_data['income'].apply(lambda x: x[:-1] if x.endswith('.') else x)

print("age", adult_data['age'].unique())
print("workclass", adult_data['workclass'].unique())
print("fnlwgt", adult_data['fnlwgt'].unique())
print("education", adult_data['education'].unique())
print("education-num", adult_data['education-num'].unique())
print("marital-status", adult_data['marital-status'].unique())
print("occupation", adult_data['occupation'].unique())
print("relationship", adult_data['relationship'].unique())
print("race", adult_data['race'].unique())
print("sex", adult_data['sex'].unique())
print("capital-gain", adult_data['capital-gain'].unique())
print("capital-loss", adult_data['capital-loss'].unique())
print("hours-per-week", adult_data['hours-per-week'].unique())
print("native-country", adult_data['native-country'].unique())
print("income", adult_data['income'].unique())

# In[11]:


# Drop unnecessary columns:
# 'education' is redundant because 'education-num' already encodes it numerically.
# 'fnlwgt' represents sampling weights and does not contribute to feature relationships.

adult_data = adult_data.drop(columns=['education', 'fnlwgt'])

# In[12]:


adult_data.head(5)

# In[13]:


categorical_columns = [
    'workclass',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]
for col in categorical_columns:
    encoder = LabelEncoder()
    not_null_idx = adult_data[col].notnull()
    adult_data.loc[not_null_idx, col] = encoder.fit_transform(adult_data.loc[not_null_idx, col])

# In[14]:


adult_data.head(5)

# In[15]:


# List of (X_index, Y_index, Z_index) triplets for CMI calculation
# Column indices after dropping 'education' and 'fnlwgt':

# 0: age
# 1: workclass
# 2: education-num
# 3: marital-status
# 4: occupation
# 5: relationship
# 6: race
# 7: sex
# 8: capital-gain
# 9: capital-loss
# 10: hours-per-week
# 11: native-country
# 12: income

cmi_triplets = [
    (0, 12, 2),  # (age, income, education-num)
    (6, 12, 2),  # (race, income, education-num)
    (7, 12, 2),  # (sex, income, education-num)
    (11, 12, 2),  # (native-country, income, education-num)
    (0, 10, 4),  # (age, hours-per-week, occupation)
    (6, 10, 4),  # (race, hours-per-week, occupation)
    (7, 10, 3),  # (sex, hours-per-week, marital-status)
    (11, 10, 4),  # (native-country, hours-per-week, occupation)
]

# In[16]:


# continuous and discrete column indices
discrete_columns = [1, 3, 4, 5, 6, 7, 11, 12]
continuous_columns = [0, 2, 8, 9, 10]

# Create a copy of the original data to preserve the raw values
adult_data_scaled = adult_data.copy()

# Change dtype to float
for col in continuous_columns:
    col_name = adult_data_scaled.columns[col]
    adult_data_scaled[col_name] = adult_data_scaled[col_name].astype(float)

# Normalize only the continuous columns using MinMaxScaler
scaler = MinMaxScaler()
adult_data_scaled.iloc[:, continuous_columns] = scaler.fit_transform(adult_data_scaled.iloc[:, continuous_columns])

# In[17]:


# Sampling strategy (Test vs Full)

# adult_data_sampled = adult_data_scaled

if sample_name:
    with open(os.path.join("samples", f"{sample_name}.txt"), "r") as f:
        line = f.read()
        sample_indices = list(map(int, line.strip().split()))
        sample_size = len(sample_indices)
else:
    sample_indices = np.random.choice(adult_data_scaled.shape[0], size=sample_size, replace=False)
    os.makedirs('samples', exist_ok=True)
    sample_name = f"sample_{run_id}_{sample_size}"
    with open(os.path.join("samples", f"{sample_name}.txt"), "w") as f:
        f.write(" ".join(map(str, sample_indices)))

adult_data_sampled = adult_data_scaled.iloc[sample_indices]
data_np = adult_data_sampled.values.astype(float)
data_tensor = torch.tensor(data_np, dtype=torch.float64).to(device)

# In[18]:


# Encode and combine data

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoded = encoder.fit_transform(adult_data_sampled.iloc[:, discrete_columns])
continuous_part = adult_data_sampled.iloc[:, continuous_columns].values
data_processed = np.concatenate([one_hot_encoded, continuous_part], axis=1)

one_hot_encoded_tensor = torch.tensor(one_hot_encoded, dtype=torch.float64).to(device)
continuous_part_tensor = torch.tensor(continuous_part, dtype=torch.float64).to(device)
data_processed_tensor = torch.cat([one_hot_encoded_tensor, continuous_part_tensor], dim=1)

# In[19]:


# GRID SERACH
# BOOTSTRAP


# In[20]:


sys.path.append(r"C:\Users\admin\Desktop\Imputation_Fairness")

# In[21]:


for triplet in cmi_triplets:
    cmi1, cmi2, cmi3 = compute_all_cmi_methods(data_tensor, data_processed_tensor, triplet, encoder, discrete_columns, continuous_columns
                                               )

    print(f"Triplet {triplet} :")
    print("  CMI (Soft Kronecker + Gaussian):", cmi1.item())
    print("  CMI (Gumbel-Softmax + Gaussian):", cmi2.item())
    print("  CMI (Gumbel on Discrete + Separate Kernels):", cmi3.item())
    print()

# # Injection of 25% MCAR Missingness

# In[22]:


# Step 1: Split the full dataset into features (X) and target (Y)
X = adult_data_sampled.iloc[:, :-1]
Y = adult_data_sampled.iloc[:, -1]

# Step 2: Inject 25% MCAR missingness into the feature set X
generator_mcar25 = Inject_Missing_Values()
X_miss_mcar25, index_mcar25 = generator_mcar25.MCAR(X, missing_rate=25)

# Step 3: Report total missing percentage
total_missing_percentage = X_miss_mcar25.isnull().sum().sum() / X_miss_mcar25.size * 100
print(f"Total Missing Percentage (MCAR 25%): {total_missing_percentage:.2f}%")

# Step 4: Report missing percentage per column
missing_per_column = (X_miss_mcar25.isnull().sum() / len(X_miss_mcar25)) * 100
print(missing_per_column)

# In[23]:

X_miss_with_Y = X_miss_mcar25.copy()
X_miss_with_Y["target"] = Y

X_numpy = X.to_numpy(dtype=float)
X_tensor = torch.tensor(X_numpy, dtype=torch.float64).to(device)

Y_numpy = Y.to_numpy(dtype=int)
Y_tensor = torch.tensor(Y_numpy, dtype=torch.int32).to(device)
X_tensor_with_Y = torch.cat([X_tensor, Y_tensor.reshape(-1, 1)], dim=1)

X_miss_mcar25_numpy = X_miss_mcar25.to_numpy(dtype=float)
X_miss_mcar25_tensor = torch.tensor(X_miss_mcar25_numpy, dtype=torch.float64).to(device)  # converting to tensor
missing_value_mask = torch.isnan(X_miss_mcar25_tensor).double().to(device)

X_miss_with_Y_numpy = X_miss_with_Y.to_numpy(dtype=float)
X_miss_with_Y_tensor = torch.tensor(X_miss_with_Y_numpy, dtype=torch.float64).to(device)
missing_value_mask_with_Y = torch.isnan(X_miss_with_Y_tensor).double().to(device)


# In[25]:


# Sinkhorn
def impute_sinkhorn():
    batchsize = 128
    lr = 1e-2
    epsilon = pick_epsilon(X_miss_mcar25_tensor)  # Determines Sinkhorn regularization strength
    sinkhorn_imputer = SinkhornImputation(eps=epsilon, batchsize=batchsize, lr=lr, niter=iterations)
    sinkhorn_filled_tensor, maes_mcar25, rmses_mcar25 = sinkhorn_imputer.fit_transform(
        X_miss_mcar25_tensor,
        verbose=True,
        report_interval=50,
        X_true=X_tensor,  # for tracking MAE/RMSE during training
    )
    return sinkhorn_filled_tensor


def impute_sinkhorn_cmi(train_triplet_index, train_cmi, highest_lambda_cmi):
    # Use train_triplet_index=-1 for all triplets
    batchsize = 128
    lr = 1e-2
    epsilon = pick_epsilon(X_miss_with_Y_tensor)
    sk_imputer = SinkhornImputation_CMI(
        eps=epsilon,
        batchsize=batchsize,
        lr=lr,
        niter=iterations,
        highest_lamda_cmi=highest_lambda_cmi,
        cmi_index=train_cmi,
    )
    sk_imp_mcar25, maes_mcar25, rmses_mcar25, history_mcar25 = sk_imputer.fit_transform(
        X_miss_with_Y_tensor,
        Y=Y_tensor if train_cmi in [1, 2] else None,
        data_processed=data_processed_tensor,
        one_hot_encoded=one_hot_encoded_tensor,
        continuous_part=continuous_part_tensor,
        verbose=True,
        report_interval=50,
        X_true=torch.cat([X_tensor, Y_tensor.reshape(-1, 1)], dim=1),  # ground truth data
        X_cols=[cmi_triplets[train_triplet_index][0]] if train_triplet_index != -1 else [t[0] for t in cmi_triplets],
        Y_cols=[cmi_triplets[train_triplet_index][1]] if train_triplet_index != -1 else [t[1] for t in cmi_triplets],
        Z_cols=[cmi_triplets[train_triplet_index][2]] if train_triplet_index != -1 else [t[2] for t in cmi_triplets],
        encoder=encoder,
        discrete_columns=discrete_columns,
        continuous_columns=continuous_columns
    )

    return sk_imp_mcar25


def impute_mean():
    mean_imp_mcar25 = SimpleImputer().fit_transform(X_miss_mcar25_tensor.cpu())
    return torch.Tensor(mean_imp_mcar25).to(device)


# Imputation by Chained Equations
def impute_mice():
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    ice_imp_mcar25 = IterativeImputer(random_state=0, max_iter=iterations).fit_transform(X_miss_mcar25_tensor.cpu())
    ice_imp_mcar25_torch = torch.tensor(ice_imp_mcar25).to(device)
    return ice_imp_mcar25_torch


# Soft Imputation
def impute_soft():
    from SoftImpute import cv_softimpute, softimpute

    cv_error_mcar25, grid_lambda_mcar25 = cv_softimpute(X_miss_mcar25_numpy, grid_len=15)
    lbda_mcar25 = grid_lambda_mcar25[np.argmin(cv_error_mcar25)]
    soft_imp_mcar25 = softimpute((X_miss_mcar25_numpy), lbda_mcar25)[1]
    soft_imp_mcar25_torch = torch.tensor(soft_imp_mcar25).to(device)
    return soft_imp_mcar25_torch


def make_processed_data(X_with_y, discrete_columns, continuous_columns):
    data_continuous = X_with_y[:, continuous_columns]
    discrete_distributions = generate_distributions_for_discrete_data(X_with_y, discrete_columns, encoder)
    data_combined_tensor = torch.cat([discrete_distributions, data_continuous], dim=1)
    return data_combined_tensor


def evaluate_imputation(imputed_tensor, missing_value_mask, ground_truth, discrete_columns, continuous_columns,
                        meta_data={}):
    # e.g, meta_data = {'imputation_method': 'sk', train_triplet: 0}
    results = []
    with torch.no_grad():
        final_mae = MAE(imputed_tensor, ground_truth, missing_value_mask).item()
        final_rmse = RMSE(imputed_tensor, ground_truth, missing_value_mask).item()
        print(f"Final MAE: {final_mae:.4f}")
        print(f"Final RMSE: {final_rmse:.4f}")
        Y_tensor_reshaped = Y_tensor.view(-1, 1)
        X_with_y = torch.cat([imputed_tensor, Y_tensor_reshaped], dim=1)
        # Estimate (CMI) after imputation
        # for each predefined triplet using all three CMI estimation methods
        for test_triplet_index, triplet in enumerate(cmi_triplets):
            processed_tensor = make_processed_data(X_with_y, discrete_columns, continuous_columns)
            cmi1, cmi2, cmi3 = compute_all_cmi_methods(X_with_y,
                                                       processed_tensor,  # One-hot encoded + scaled continuous
                                                       triplet,  # (X, Y, Z)
                                                       encoder,
                                                       discrete_columns,
                                                       continuous_columns
                                                       )
            print(f"[{meta_data.get('imputation_method', 'Undefined Method')}] Triplet {triplet} :")
            print(" CMI (Soft Kronecker + Gaussian):", f"{cmi1.item():.4f}")
            print(" CMI (Gumbel-Softmax + Gaussian):", f"{cmi2.item():.4f}")
            print(" CMI (Gumbel on Discrete + Separate Kernels):", f"{cmi3.item():.4f}")
            print()
            for cmi_idx, cmi in enumerate([cmi1, cmi2, cmi3]):
                results.append({
                    **meta_data,
                    'test_triplet': test_triplet_index + 1,
                    'MAE': final_mae,
                    'RMSE': final_rmse,
                    'cmi_method': f'method-{cmi_idx + 1}',
                    'cmi': cmi.item(),
                })
    results_df = pd.DataFrame(results)
    return results_df


result_df = pd.DataFrame()

if 'sk_cmi' in imp_methods:
    for train_triplet_index in [-1] + list(range(len(cmi_triplets))):
        for train_cmi in range(3):
            imputed_tensor = impute_sinkhorn_cmi(train_triplet_index, train_cmi, highest_lambda_cmi=500)
            df = evaluate_imputation(imputed_tensor,
                                     missing_value_mask_with_Y,
                                     X_tensor_with_Y,
                                     discrete_columns,
                                     continuous_columns,
                                     meta_data={
                                         'imputation_method': 'SinkHornCMI',
                                         'train_cmi_type': int(train_cmi) + 1,
                                         'train_triplet': train_triplet_index + 1,
                                     })
            result_df = pd.concat([result_df, df], ignore_index=True)

if 'sk_cmi0' in imp_methods:
    for train_triplet_index in [-1] + list(range(len(cmi_triplets))):
        imputed_tensor = impute_sinkhorn_cmi(train_triplet_index, train_cmi=0, highest_lambda_cmi=0)
        df = evaluate_imputation(imputed_tensor,
                                 missing_value_mask_with_Y,
                                 X_tensor_with_Y,
                                 discrete_columns,
                                 continuous_columns,
                                 meta_data={
                                     'imputation_method': 'SinkHornCMI0',
                                     'train_triplet': train_triplet_index + 1,
                                 })
        result_df = pd.concat([result_df, df], ignore_index=True)


if 'sk' in imp_methods:
    imputed_tensor = impute_sinkhorn()
    df = evaluate_imputation(imputed_tensor, missing_value_mask, X_tensor, discrete_columns, continuous_columns,
                             meta_data={'imputation_method': 'SinkHorn', })
    result_df = pd.concat([result_df, df], ignore_index=True)


if 'mean' in imp_methods:
    imputed_tensor = impute_mean()
    df = evaluate_imputation(
        imputed_tensor,
        missing_value_mask,
        X_tensor,
        discrete_columns,
        continuous_columns,
        meta_data={'imputation_method': 'Mean', }
    )
    result_df = pd.concat([result_df, df], ignore_index=True)

if 'mice' in imp_methods:
    imputed_tensor = impute_mice()
    df = evaluate_imputation(
        imputed_tensor,
        missing_value_mask,
        X_tensor,
        discrete_columns,
        continuous_columns,
        meta_data={'imputation_method': 'Mice', })
    result_df = pd.concat([result_df, df], ignore_index=True)

if 'soft' in imp_methods:
    imputed_tensor = impute_soft()
    df = evaluate_imputation(
        imputed_tensor,
        missing_value_mask,
        X_tensor,
        discrete_columns,
        continuous_columns,
        meta_data={'imputation_method': 'Soft', })
    result_df = pd.concat([result_df, df], ignore_index=True)

os.makedirs('results', exist_ok=True)
result_df.to_csv(os.path.join('results', f'{run_id}_results.csv'), index=False)

meta_data = {
    'run_id': run_id,
    'imp_methods': imp_methods,
    'sample_name': sample_name,
    'sample_size': sample_size,
    'num_iterations': iterations,
}

with open(os.path.join('results', f'meta_{run_id}.json'), 'w') as f:
    json.dump(meta_data, f)
