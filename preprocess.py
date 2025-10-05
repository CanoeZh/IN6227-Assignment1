import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

COLUMNS = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]

DATA_DIR = Path(__file__).resolve().parents[1] / "census+income"
TRAIN_PATH = DATA_DIR / "adult.data"
TEST_PATH  = DATA_DIR / "adult.test"

def load_data():
    train = pd.read_csv(TRAIN_PATH, names=COLUMNS, skipinitialspace=True, na_values='?')
    test  = pd.read_csv(TEST_PATH,  names=COLUMNS, skipinitialspace=True, na_values='?')
    test['income'] = test['income'].astype(str).str.replace('.', '', regex=False).str.strip()
    return train, test

def preprocess(train, test):
    train = train.copy()
    test = test.copy()

    # Encode target
    train['income'] = train['income'].map({'<=50K':0, '>50K':1})
    test['income']  = test['income'].map({'<=50K':0, '>50K':1})

    X_train, y_train = train.drop('income', axis=1), train['income']
    X_test, y_test = test.drop('income', axis=1), test['income']

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric, num_cols),
        ('cat', categorical, cat_cols)
    ])

    return preprocessor, X_train, X_test, y_train, y_test
