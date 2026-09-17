import os
import time
import pandas as pd
import lingam
import numpy as np
import networkx as nx
from lingam.utils import make_dot
from typing import List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from causallearn.search.ConstraintBased.PC import pc

def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a dataset from the specified file path.
    Assumes the dataset is in CSV format.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame or None: The loaded dataframe, or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to load {file_path}. Reason: {e}")
        return None

def row_to_timeseries(df: pd.Series) -> pd.DataFrame:
    """
    Converts a single row of a DataFrame into a time series DataFrame.
    
    Args:
        df (pd.Series): A single row from the original DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame with one column representing the time series.
    """
    

def perform_causal_discovery(df: pd.DataFrame) -> lingam.VARLiNGAM:
    """
    Performs causal discovery using VARLiNGAM on the provided DataFrame.
    
    Args:
        df (pd.DataFrame): The input time series dataset.
        
    Returns:
        lingam.VARLiNGAM: The fitted VARLiNGAM model.
    """
    model = lingam.VARLiNGAM()
    model.fit(df)
    return model

def save_causal_graph(model: lingam.VARLiNGAM, labels: List[str], output_path: str):
    """
    Saves the causal graph as a .dot file.
    
    Args:
        model (lingam.VARLiNGAM): The fitted VARLiNGAM model.
        labels (List[str]): List of variable names.
        output_path (str): The file path where the .dot file will be saved.
    """
    try:
        # For VARLiNGAM, adjacency_matrices_ is a list of matrices for each lag.
        # make_dot can handle the list of matrices to visualize contemporaneous and lagged effects.
        dot = make_dot(np.hstack(model.adjacency_matrices_), ignore_shape=True, labels=(labels * len(model.adjacency_matrices_)))
        
        # Write the Graphviz source directly to a .dot file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dot.source)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: make_dot with adjacency_matrices_ failed ({e}). Trying with instantaneous effects only (lag 0).")
        print(f"Length of adjacency_matrices[0]: {len(model.adjacency_matrices_[0])}\nLength of labels: {len(labels)}")
        try:
            # Fallback to contemporaneous effects if the list is not supported
            dot = make_dot(model.adjacency_matrices_[0], labels=labels)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dot.source)
        except Exception as e_inner:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to save .dot file to {output_path}. Reason: {e_inner}")

def impute_and_encode_features(df: pd.DataFrame):
    """Imputes missing values (median for numeric, mode for categorical)

    and one-hot encodes categorical columns for numeric compatibility.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                SimpleImputer(strategy="most_frequent"),
                categorical_cols,
            )
        )

    if not transformers:
        return pd.DataFrame()

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    imputed_array = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    df_imputed = pd.DataFrame(
        imputed_array, columns=feature_names, index=df.index
    )

    if categorical_cols:
        df_imputed = pd.get_dummies(
            df_imputed, columns=categorical_cols, drop_first=True
        )

    return df_imputed.apply(pd.to_numeric), df_imputed.columns


def fit_varlingam(
    df: pd.DataFrame, df_name: str
) -> Tuple[Optional[object], float]:
    """Runs VARLiNGAM causal discovery and returns the fitted model and elapsed time."""
    print(
        f"[{time.strftime('%H:%M:%S')}] INFO: Running VARLiNGAM causal discovery..."
    )
    start_time = time.time()

    try:
        model = perform_causal_discovery(df)
        elapsed_time = time.time() - start_time
        print(
            f"[{time.strftime('%H:%M:%S')}] INFO: Causal discovery completed in {elapsed_time:.2f} seconds."
        )
        return model, elapsed_time
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S')}] ERROR: VARLiNGAM fitting failed for {df_name}. Reason: {e}"
        )
        return None, 0.0


def export_causal_graph(
    model: object, labels: list[str], df_name: str
) -> None:
    """Exports the discovered model to a Graphviz .dot file."""
    base_name = os.path.splitext(os.path.basename(df_name))[0]
    output_dot_path = f"{base_name}_causal_graph.dot"

    print(
        f"[{time.strftime('%H:%M:%S')}] INFO: Saving causal graph to '{output_dot_path}'..."
    )
    save_causal_graph(model, labels, output_dot_path)


def check_and_extract_cycles(
    varlingam_model,
    variable_names: list[str] | None = None,
    include_lags: bool = True,
    threshold: float = 1e-5,
) -> tuple[bool, list[list[str]]]:
    """
    Checks if a VAR-LiNGAM result contains cycles and extracts them.
    """

    # Extract adjacency matrices
    adj_matrices = varlingam_model.adjacency_matrices_

    # In VAR-LiNGAM, adj_matrices has shape (n_lags + 1, n_features, n_features)
    # where adj_matrices[0] is B0 (lag 0) and adj_matrices[tau] is B_tau
    n_vars = adj_matrices.shape[1]
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_vars)]

    G = nx.DiGraph()
    G.add_nodes_from(variable_names)

    matrices_to_evaluate = adj_matrices if include_lags else [adj_matrices[0]]

    # VAR-LiNGAM convention: row i is target, col j is source (x_j -> x_i)
    for B in matrices_to_evaluate:
        rows, cols = np.where(np.abs(B) > threshold)
        for target_idx, source_idx in zip(rows, cols):
            if target_idx != source_idx:  # Skip self-loops if considering summary graph
                G.add_edge(variable_names[source_idx], variable_names[target_idx])

    is_dag = nx.is_directed_acyclic_graph(G)
    cycles = list(nx.simple_cycles(G)) if not is_dag else []

    return is_dag, cycles


def process_single_dataset(df: pd.DataFrame, df_name: str) -> None:
    """Orchestrates dataset preprocessing, causal discovery, and artifact saving."""
    print(
        f"[{time.strftime('%H:%M:%S')}] INFO: --- Starting processing for dataset '{df_name}' ---"
    )

    if df.empty:
        print(
            f"[{time.strftime('%H:%M:%S')}] WARNING: Dataset '{df_name}' is initially empty. Skipping."
        )
        return

    # 1. Preprocess & Impute
    df_processed, new_columns = impute_and_encode_features(df)
    if df_processed.empty or df_processed.shape[1] < 2:
        print(
            f"[{time.strftime('%H:%M:%S')}] WARNING: Dataset '{df_name}' has insufficient numeric data after preprocessing. Skipping."
        )
        return

    # 2. Causal Discovery
    model, _ = fit_varlingam(df_processed, df_name)
    if model is None:
        return

    # 2.5. Check if graph is DAG
    print("Checking if the graph is DAG...")

    try:
        is_dag, cycles = check_and_extract_cycles(model, list(df.columns), True)
        print(is_dag)

    except Exception as e:
        print("ERROR: ", e)

    # 3. Export
    labels = new_columns.to_list()
    output_dot_path = f"{df_name}_causal_graph.dot"
    save_causal_graph(model, labels, output_dot_path)

    print(
        f"[{time.strftime('%H:%M:%S')}] INFO: --- Successfully processed dataset '{df_name}' ---\n"
    )

def run_pipeline(datasets):
    """
    Runs the complete causal discovery pipeline on a list of file names.
    
    Args:
        datasets: A list of pairs, where the first element is the dataset and the second is its name.
    """
    if not datasets:
        print("No datasets provided to the pipeline.")
        return

    print("==================================================")
    print(f"Starting Causal Discovery Pipeline for {len(datasets)} dataset(s).")
    print("==================================================\n")
    
    for df, df_name in datasets:
        if df is not None:
            process_single_dataset(df, df_name)

    print("==================================================")
    print("Pipeline execution completed.")
    print("==================================================")

if __name__ == "__main__":
    # Example usage:
    # Provide the list of your dataset file paths below
    datasets = []

    df_6 = load_dataset("cleaned_data/trial6.csv")
    df_6 = df_6.drop(columns=['YS_delta_cystatin_c_24m', 
                              'YS_delta_1_5ag_24m', 
                              'YS_delta_1_5ag_12m'
                              ])
    datasets.append((df_6, "trial6.csv"))

    df_29 = load_dataset("cleaned_data/trial29.csv")
    df_29 = df_29.drop(columns=['X_csf_qcc_0w'])
    datasets.append((df_29, "trial29.csv"))

    df_37 = load_dataset("cleaned_data/trial37.csv")
    df_37 = df_37.drop(columns=['X_pdstent_0d',
                                'X_sodsom_0d',
                                'X_bsphinc_0d',
                                'X_chole_0d'
                                ])
    datasets.append((df_37, "trial37.csv"))

    df_116 = load_dataset("cleaned_data/trial116.csv")
    df_116 = df_116.drop(columns=['X_source_id', 
                                  'YP_dass_total_t2', 
                                  'YP_delta_dass_total_t2', 
                                  'YP_delta_factg_total_t2', 
                                  'YS_dass_stress_t2', 
                                  'YS_dass_anxiety_t2', 
                                  'YS_dass_depression_t2', 
                                  'X_dass_total_0m', 
                                  'X_age_years', 
                                  'X_residence_place_code', 
                                  'X_marital_status_code', 
                                  'X_income_code', 
                                  'X_treatment_type_code',
                                  'X_cancer_diagnosis_code'
                                  ])
    datasets.append((df_116, "trial116.csv"))
    print(f"colunas em 116: {len(df_116.columns)}")

    dfs_to_load = [
            "cleaned_data/trial13.csv",
            "cleaned_data/trial108.csv",
            "cleaned_data/trial120.csv",
        ]

    for path in dfs_to_load:
        df = load_dataset(path)
        if df is not None:
            datasets.append((df, os.path.basename(path)))

    if datasets:
        run_pipeline(datasets)
    else:
        print("Please provide a list of datasets in the 'datasets' list to run the pipeline.")
