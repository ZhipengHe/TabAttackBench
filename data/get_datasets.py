import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
from scipy.io import arff
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from itertools import chain

from .data_config import DATASET_INFO

@dataclass
class DfInfo:

    # df_info.scaler.data_min_ = [] # df_info.numerical_cols
    # df_info.scaler.data_max_ = [] # (0,1)
    # inverse_dummy
    # inverse_scaling
    # inverse_scaling_and_dummy

    ## Original data frame
    df: pd.DataFrame 

    ## All feature names
    feature_names: List

    ###### `numerical_cols` and `categorical_cols` may contain column name of target.
    
    ## All numerical columns
    numerical_cols: List

    ## All categorical columns
    categorical_cols: List


    num_categories_list: Optional[List[int]]

    ## type of each columns
    # columns_type: Dict

    ## Label(target) column name
    target_name: str

    ## Unique values in the target column.
    possible_outcomes: List 

    ## Dataframe with the numerical columns scaled by MinMaxScaler to [0, 1]
    scaled_df: pd.DataFrame

    ## MinMaxScaler to scale numerical columns. ()
    scaler: Any

    ## Dictionary {"categorical_col_name": "all of its ohe column names"}  
    cat_to_ohe_cat: Dict

    ## All feature names in ohe format
    ohe_feature_names: List

    ## LabelEncoder used to encoding the target column.
    target_label_encoder: Any

    ## Dataframe with scaled numerical features and dummy categorical features (The dataframe used for training).
    dummy_df: pd.DataFrame


    def get_ohe_cat_cols(self,):
        return list(chain(*[ v for v in self.cat_to_ohe_cat.values()]))

    def get_ohe_num_cols(self,):
        return self.numerical_cols

    def get_numerical_mads(self,):
        return self.scaled_df[self.numerical_cols].mad().to_dict()

    def get_numerical_stds(self,):
        return self.scaled_df[self.numerical_cols].std().to_dict()

def get_columns_type(df):
    '''
    Identify the column types to later classify them as categorical or numerical columns (features).
    '''

    # And here, we include both 64 bytes and 32 bytes.
    integer_features = list(df.select_dtypes(include=['int64']).columns) +  list(df.select_dtypes(include=['int32']).columns)
    float_features = list(df.select_dtypes(include=['float64']).columns) + list(df.select_dtypes(include=['float32']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    columns_type = {
        'integer': integer_features,
        'float': float_features,
        'string': string_features,
    }

    numerical_cols = columns_type['integer'] + columns_type['float']
    categorical_cols = columns_type['string']

    return numerical_cols, categorical_cols, columns_type

def transform_to_dummy(df, categorical_cols):
    '''
    Tranform the categorical columns to ohe format.
    '''
    ## For nerual network, we feed in the one-hot encoded vector.
    for col in categorical_cols:
        df = pd.concat([df,pd.get_dummies(df[col], prefix=col)],axis=1)
        df.drop([col],axis=1, inplace=True)
    return df

def min_max_scale_numerical(df, numerical_cols):
    '''
    Scale the numerical columns in the dataframe.
    '''

    ## Scaling the numerical data.
    scaled_df = df.copy(deep=True)
    scaler = MinMaxScaler()
    scaled_df[numerical_cols] = scaler.fit_transform(scaled_df[numerical_cols])
    return scaled_df, scaler

def inverse_dummy(dummy_df, cat_to_ohe_cat):
    '''
    Inverse the process of pd.get_dummy().
    [`cat_to_ohe_cat`] -> Dictionary `{"column_name": list("ohe_column_name")}`
    '''
    not_dummy_df = dummy_df.copy(deep=True)
    for k in cat_to_ohe_cat.keys():
        not_dummy_df[k] = dummy_df[cat_to_ohe_cat[k]].idxmax(axis=1)
        not_dummy_df[k] = not_dummy_df[k].apply(lambda x: x.replace(f'{k}_',""))
        not_dummy_df.drop(cat_to_ohe_cat[k], axis=1, inplace=True)
    return not_dummy_df

def inverse_scaling(scaled_df, df_info):
    result_df = scaled_df.copy(deep=True)

    result_df[df_info.numerical_cols] = df_info.scaler.inverse_transform(
                    result_df[df_info.numerical_cols])

    return result_df

def inverse_scaling_and_dummy(scaled_dummy_df, df_info):
    return inverse_scaling(inverse_dummy(scaled_dummy_df, df_info.cat_to_ohe_cat), df_info)


def get_cat_ohe_info(dummy_df, categorical_cols, target_name):
    '''
    Get ohe informatoin required for counterfactual generator (DiCE, Alibi) to recognise categorical features. 
    '''

    cat_to_ohe_cat = {}
    for c_col in categorical_cols:
        if c_col != target_name:
            cat_to_ohe_cat[c_col] = [ ohe_col for ohe_col in dummy_df.columns if ohe_col.startswith(c_col) and ohe_col != target_name]

    ohe_feature_names = [ col for col in dummy_df.columns if col != target_name]

    return cat_to_ohe_cat, ohe_feature_names


def remove_missing_values(df):
    '''
    Remove the rows with missing value in the dataframe.
    '''
        # Handle missing values
    # Remove features with more than 50% missing values
    threshold = 0.5 * len(df)
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # df = pd.DataFrame(df.to_dict())
    for col in df.columns:
        if '?' in list(df[col].unique()):
            ### Replace the missing value by the most common.
            df.loc[df[col] == '?', col] = df[col].value_counts().index[0]
            
    return df




def load_dataset(dataset_name):
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Fetch the dataset from the URL
    url = DATASET_INFO[dataset_name]["url"]
    column_names = DATASET_INFO[dataset_name]["column_names"]

    # Determine file format based on the URL's file extension
    file_extension = os.path.splitext(url)[-1].lower()
    local_file_path = os.path.join("data", "datasets", dataset_name + file_extension)

    # Specify the directory path you want to create if it doesn't exist
    directory_path = "data/datasets"

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)

    if file_extension in ['.csv', '.data']:
        # Check if the file exists locally
        if not os.path.isfile(local_file_path):
            response = requests.get(url)
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
        
        data = pd.read_csv(local_file_path, header=None, names=column_names, 
                           sep=',\s*', engine='python')

    elif file_extension == '.arff':
        if not os.path.isfile(local_file_path):
            response = requests.get(url)
            with open(local_file_path, 'wb') as f:
                f.write(response.content)

        data, meta = arff.loadarff(local_file_path)

        data = pd.DataFrame(data)
        
    else:
        raise ValueError(f"Unsupported file format for URL: {url}")
    

    return data, column_names



def get_dataclass(dataset_name):

    data, column_names = load_dataset(dataset_name)

    target_col = DATASET_INFO[dataset_name]["target_col"]

    data = remove_missing_values(data)

    possible_outcomes = data[target_col].unique().tolist()

    # # Apply target_transform lambda function if available
    if DATASET_INFO[dataset_name]["target_transform"] is not None:
        data[target_col] = data[target_col].apply(DATASET_INFO[dataset_name]["target_transform"])

    # Check for and keep columns with more than one unique value
    columns_with_values = []
    for col in data.columns:
        if data[col].nunique() > 1:
            columns_with_values.append(col)
    data = data[columns_with_values]


    # Define categorical and numerical columns
    categorical_cols = [col for col in DATASET_INFO[dataset_name]["categorical_cols"] if col in columns_with_values]
    numerical_cols = [col for col in DATASET_INFO[dataset_name]["numerical_cols"] if col in columns_with_values]


    # Create a dictionary to store the number of unique categories for each categorical column
    categories = {}

    # Calculate the number of unique categories for each categorical column
    for col in categorical_cols:
        unique_values = data[col].nunique()
        if not pd.isnull(unique_values):
            categories[col] = unique_values

    # Convert the dictionary to a list of values
    num_categories_list = list(categories.values())

    # # Use LabelEncoder to encode categorical columns
    # label_encoders = {}
    # for col in categorical_cols:
    #     le = LabelEncoder()
    #     data[col] = le.fit_transform(data[col])
    #     label_encoders[col] = le


    # Apply MinMacScaler [0, 1] to numerical columns
    if len(numerical_cols) > 0:
        scaled_df, scaler = min_max_scale_numerical(data, numerical_cols)
    else:
        scaled_df = data
        scaler = None


    # Get one-hot encoded features.
    dummy_df = pd.get_dummies(
        scaled_df,
        columns=[col for col in categorical_cols],
        #   drop_first=True
        )
    
    ## Get one-hot encoded info
    cat_to_ohe_cat, ohe_feature_names = get_cat_ohe_info(dummy_df, categorical_cols, target_col)


    ## Preprocessing the label
    target_label_encoder = LabelEncoder()
    dummy_df[target_col] = target_label_encoder.fit_transform(dummy_df[target_col])


    dummy_df= dummy_df[ohe_feature_names + [target_col]]

    bool_columns = dummy_df.select_dtypes(include=bool).columns
    dummy_df[bool_columns] = dummy_df[bool_columns].astype(int)

    return DfInfo(data, column_names, numerical_cols, categorical_cols, num_categories_list, target_col, possible_outcomes, scaled_df, scaler, cat_to_ohe_cat, ohe_feature_names, target_label_encoder, dummy_df)


def get_split(dataset_name, device):

    df_info = get_dataclass(dataset_name)

    train_df, test_df = train_test_split(
        df_info.dummy_df, test_size=0.2, random_state=42, shuffle=True
    )
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, shuffle=True)

    X_train = np.array(train_df[df_info.ohe_feature_names])
    y_train = np.array(train_df[df_info.target_name])
    X_val = np.array(val_df[df_info.ohe_feature_names])
    y_val = np.array(val_df[df_info.target_name])
    X_test = np.array(test_df[df_info.ohe_feature_names])
    y_test = np.array(test_df[df_info.target_name])


    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)


    # Return the training and test sets
    return X_train, y_train, X_val, y_val, X_test, y_test, \
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor, \
        df_info

def get_split_continues(dataset_name, device):

    df_info = get_dataclass(dataset_name)

    train_df, test_df = train_test_split(
        df_info.dummy_df, test_size=0.2, random_state=42, shuffle=True
    )
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, shuffle=True)

    X_train = np.array(train_df[df_info.numerical_cols])
    y_train = np.array(train_df[df_info.target_name])
    X_val = np.array(val_df[df_info.numerical_cols])
    y_val = np.array(val_df[df_info.target_name])
    X_test = np.array(test_df[df_info.numerical_cols])
    y_test = np.array(test_df[df_info.target_name])


    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)


    # Return the training and test sets
    return X_train, y_train, X_val, y_val, X_test, y_test, \
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor, \
        df_info


if __name__ == "__main__":
    import pickle

    # Test the get_dataset function
    lists = ["Adult", "Electricity", "Higgs", "Mushroom"] # "Adult", "Electricity", "Higgs", "KDDCup09_appetency", "Mushroom"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data in lists:
        X_train, y_train, X_val, y_val, X_test, y_test, \
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor, \
            info = get_split(data, device)
        
        # Save the data and info to a local file
        output_filename = f"data/datasets/{data}_processed.pkl"
        with open(output_filename, 'wb') as file:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test,
                         X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                         X_val_tensor, y_val_tensor, info), file)

        print("#############################################")
        # print(f'{info=}')
        print(f'{X_train.shape=}')
        print(f'{y_train.shape=}')
        print(f'{X_val.shape=}')
        print(f'{y_val.shape=}')
        print(f'{X_test.shape=}')
        print(f'{y_test.shape=}')
        print(f'{X_train_tensor.shape=}')
        print(f'{y_train_tensor.shape=}')
        print(f'{X_test_tensor.shape=}')
        print(f'{y_test_tensor.shape=}')
        print(f'{X_val_tensor.shape=}')
        print(f'{y_val_tensor.shape=}')

        print("#############################################")


