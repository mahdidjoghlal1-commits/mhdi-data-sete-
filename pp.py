import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# Étape 1 : Lecture des données (READ)
# ============================================================

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

print(f"\n✅ {len(columns)} colonnes définies")

train_path = "dataset/KDDTrain+.txt"
df_train = pd.read_csv(train_path, names=columns, sep=',', engine='python')
print(f"✅ Train : {len(df_train)} lignes × {len(df_train.columns)} colonnes")

test_path = "dataset/KDDTest+.txt"
df_test = pd.read_csv(test_path, names=columns, sep=',', engine='python')
print(f"✅ Test : {len(df_test)} lignes × {len(df_test.columns)} colonnes")

print("\n📊 5 premières lignes de Train :")
print(df_train.head())

print("\n📊 5 premières lignes de Test :")
print(df_test.head())

print("\n🏷️ Distribution des labels dans Train :")
print(df_train['label'].value_counts())

print("\n🏷️ Distribution des labels dans Test :")
print(df_test['label'].value_counts())

print("\n" + "=" * 60)
print("📋 Résumé de l'étape 1 :")
print("=" * 60)
print(f"   Train : {df_train.shape[0]} lignes, {df_train.shape[1]} colonnes")
print(f"   Test : {df_test.shape[0]} lignes, {df_test.shape[1]} colonnes")

normal_train = (df_train['label'] == 'normal').sum()
normal_test = (df_test['label'] == 'normal').sum()

print(f"   🏷️  normal dans Train : {normal_train}")
print(f"   🏷️  normal dans Test : {normal_test}")

print("\n✅ Étape 1 terminée avec succès !")

# ============================================================
# Étape 2 : Nettoyage des données
# ============================================================

missing_train = df_train.isnull().sum()
missing_test = df_test.isnull().sum()

missing_train_cols = missing_train[missing_train > 0]
missing_test_cols = missing_test[missing_test > 0]

print("📊 Détection des valeurs manquantes :")
print(f"   Train : {len(missing_train_cols)} colonnes avec valeurs manquantes")
print(f"   Test : {len(missing_test_cols)} colonnes avec valeurs manquantes")

cols_all_null_train = df_train.columns[df_train.isnull().all()].tolist()
cols_all_null_test = df_test.columns[df_test.isnull().all()].tolist()

if cols_all_null_train:
    df_train = df_train.drop(columns=cols_all_null_train)

if cols_all_null_test:
    df_test = df_test.drop(columns=cols_all_null_test)

numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df_train[col].isnull().sum() > 0:
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)

dup_train_before = df_train.duplicated().sum()
dup_test_before = df_test.duplicated().sum()

if dup_train_before > 0:
    df_train = df_train.drop_duplicates()

if dup_test_before > 0:
    df_test = df_test.drop_duplicates()

const_cols = []
for col in df_train.columns:
    if col != 'label':
        if df_train[col].nunique() == 1:
            const_cols.append(col)

if const_cols:
    df_train = df_train.drop(columns=const_cols)
    df_test = df_test.drop(columns=const_cols)

df_train.to_csv("dataset/train_cleaned.csv", index=False)
df_test.to_csv("dataset/test_cleaned.csv", index=False)

print("🎯 Étape 2 terminée !")

# ============================================================
# Étape 2.5 : Encoding
# ============================================================

cat_cols = df_train.select_dtypes(include=['object']).columns.drop('label')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_train_cat = encoder.fit_transform(df_train[cat_cols])
X_test_cat = encoder.transform(df_test[cat_cols])

joblib.dump(encoder, "dataset/onehot_encoder.pkl")

y_train = (df_train['label'] != 'normal').astype(int)
y_test = (df_test['label'] != 'normal').astype(int)

print("✅ Encoding terminé")

# ============================================================
# Étape 3 : Scaling
# ============================================================

num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.drop(['difficulty'])

X_train_num = df_train[num_cols]
X_test_num = df_test[num_cols]

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

joblib.dump(scaler, "dataset/scaler.pkl")

print("✅ Scaling terminé")

# ============================================================
# Étape 4 : Fusion
# ============================================================

X_train_final = np.hstack((X_train_num_scaled, X_train_cat))
X_test_final = np.hstack((X_test_num_scaled, X_test_cat))

print(f"✅ X_train_final shape: {X_train_final.shape}")
print(f"✅ X_test_final shape: {X_test_final.shape}")

# ============================================================
# Sauvegarde
# ============================================================

np.save("dataset/X_train.npy", X_train_final)
np.save("dataset/X_test.npy", X_test_final)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

print("🎯 Dataset prêt pour ML !")