"""
mule_preprocessor.py
--------------------
Defines the MuleAccountPreprocessor class for the Bank of India Mule Account Detection pipeline.
Encapsulates date parsing, column classification, imputation, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

class MuleAccountPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, ref_date='2025-12-31'):
        self.ref_date = pd.to_datetime(ref_date)
        self.categorical_cols = ['F3886', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
        self.continuous_cols = []
        self.binary_cols = []
        
        self.imputer = None
        self.encoder = None
        self.scaler = None
        
        self.imputed_col_names = []
        self.encoded_feature_names = []
        self.remaining_continuous = []
        
        # Dropping and Selection lists (set after analysis)
        self.redundant_cols = []
        self.selected_features = []

    def fit(self, X, y=None):
        X_temp = X.copy()
        
        # Reset lists to prevent duplicates on multiple fit calls
        self.continuous_cols = []
        self.binary_cols = []
        self.categorical_cols = ['F3886', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
        
        # 1. Perform Date conversion first to get the columns for typing
        if 'F3888' in X_temp.columns:
            opening_date = pd.to_datetime(X_temp['F3888'], format='mixed', errors='coerce')
            X_temp['account_age_days'] = (self.ref_date - opening_date).dt.days
            X_temp['account_age_years'] = X_temp['account_age_days'] / 365.25
            X_temp = X_temp.drop(columns=['F3888'])
            
        # 2. Separate numerical columns into binary and continuous
        numeric_cols = [c for c in X_temp.columns if c not in self.categorical_cols]
        for col in numeric_cols:
            non_null_vals = X_temp[col].dropna().unique()
            if len(non_null_vals) <= 2:
                self.binary_cols.append(col)
            else:
                self.continuous_cols.append(col)
                
        # 3. Fit Imputer
        # Setup ColumnTransformer to apply correct strategies
        self.imputer = ColumnTransformer(transformers=[
            ('continuous', SimpleImputer(strategy='median'), self.continuous_cols),
            ('binary', SimpleImputer(strategy='most_frequent'), self.binary_cols),
            ('categorical', SimpleImputer(strategy='most_frequent'), self.categorical_cols)
        ], remainder='drop')
        
        self.imputer.fit(X_temp)
        
        # Keep track of column names after imputation
        self.imputed_col_names = self.continuous_cols + self.binary_cols + self.categorical_cols
        
        # To fit subsequent steps, we transform the training data
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_temp), 
            columns=self.imputed_col_names, 
            index=X_temp.index
        )
        
        # Cast numerical columns back to numeric
        for col in self.continuous_cols + self.binary_cols:
            X_imputed[col] = pd.to_numeric(X_imputed[col])
            
        # 4. Fit One-Hot Encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X_imputed[self.categorical_cols])
        
        # Generate readable feature names
        feature_prefix_map = {
            'F3886': 'account_type',
            'F3889': 'historical_code',
            'F3890': 'area_category',
            'F3891': 'occupation',
            'F3892': 'gender',
            'F3893': 'customer_segment'
        }
        
        self.encoded_feature_names = []
        for i, col in enumerate(self.categorical_cols):
            cats = self.encoder.categories_[i]
            prefix = feature_prefix_map.get(col, col.lower())
            for cat in cats:
                clean_cat = str(cat).lower().strip().replace(' ', '_').replace('-', '_').replace('/', '_')
                self.encoded_feature_names.append(f"{prefix}_{clean_cat}")
                
        # To fit scaler, apply one-hot encoding
        encoded_cats = self.encoder.transform(X_imputed[self.categorical_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=self.encoded_feature_names, index=X_temp.index)
        
        X_encoded = X_imputed.drop(columns=self.categorical_cols)
        X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        
        # 5. Fit Scaler (RobustScaler on continuous columns)
        # Drop redundant features during fit if they are set
        if self.redundant_cols:
            X_encoded = X_encoded.drop(columns=[c for c in self.redundant_cols if c in X_encoded.columns])
            
        # Find which continuous columns are still present
        self.remaining_continuous = [c for c in self.continuous_cols if c in X_encoded.columns]
        self.scaler = RobustScaler()
        self.scaler.fit(X_encoded[self.remaining_continuous])
        
        return self

    def transform(self, X):
        X_temp = X.copy()
        
        # 1. Perform Date conversion
        if 'F3888' in X_temp.columns:
            opening_date = pd.to_datetime(X_temp['F3888'], format='mixed', errors='coerce')
            X_temp['account_age_days'] = (self.ref_date - opening_date).dt.days
            X_temp['account_age_years'] = X_temp['account_age_days'] / 365.25
            X_temp = X_temp.drop(columns=['F3888'])
            
        # 2. Apply Imputer
        imputed_arr = self.imputer.transform(X_temp)
        X_imputed = pd.DataFrame(
            imputed_arr, 
            columns=self.imputed_col_names, 
            index=X_temp.index
        )
        
        # Cast numerical columns back to numeric
        for col in self.continuous_cols + self.binary_cols:
            X_imputed[col] = pd.to_numeric(X_imputed[col])
            
        # 3. Apply One-Hot Encoder
        encoded_cats = self.encoder.transform(X_imputed[self.categorical_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=self.encoded_feature_names, index=X_temp.index)
        
        X_encoded = X_imputed.drop(columns=self.categorical_cols)
        X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        
        # 4. Drop redundant features (if any are set)
        if self.redundant_cols:
            X_encoded = X_encoded.drop(columns=[c for c in self.redundant_cols if c in X_encoded.columns])
            
        # 5. Apply Scaler
        # Find which continuous columns are still present
        self.remaining_continuous = [c for c in self.continuous_cols if c in X_encoded.columns]
        X_scaled = X_encoded.copy()
        if self.remaining_continuous:
            X_scaled[self.remaining_continuous] = self.scaler.transform(X_encoded[self.remaining_continuous])
            
        # 6. Keep only final selected features (if set)
        if self.selected_features:
            final_cols = [c for c in self.selected_features if c in X_scaled.columns]
            # If some selected features are missing (which shouldn't happen), print warning
            missing_cols = set(self.selected_features) - set(X_scaled.columns)
            if missing_cols:
                print(f"Warning: {len(missing_cols)} selected features are missing in transform: {list(missing_cols)[:5]}...")
            return X_scaled[final_cols]
            
        return X_scaled
