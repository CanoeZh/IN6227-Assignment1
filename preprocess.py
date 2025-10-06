import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

COLUMNS = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]

DATA_DIR = Path(__file__).resolve().parent / "census+income"
TRAIN_PATH = DATA_DIR / "adult.data"
TEST_PATH  = DATA_DIR / "adult.test"

def load_data():
    train = pd.read_csv(TRAIN_PATH, names=COLUMNS, na_values='?', skipinitialspace=True)
    test  = pd.read_csv(TEST_PATH,  names=COLUMNS, na_values='?', skipinitialspace=True, skiprows=1)
    train['income'] = train['income'].str.strip()
    test['income']  = test['income'].str.strip().str.replace('.', '', regex=False)
    df = pd.concat([train, test], ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    y = (df['income'] == '>50K').astype(int)
    X = df.drop(columns=['income'])
    return X, y

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False))
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric, num_cols),
        ('cat', categorical, cat_cols)
    ], remainder='drop', sparse_threshold=1.0)
    return preprocessor

def preprocess(test_size=0.2, random_state=42):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = build_preprocessor(X)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    return preprocessor, X_train_t, X_test_t, y_train, y_test