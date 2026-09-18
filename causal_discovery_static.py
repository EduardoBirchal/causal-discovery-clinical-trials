import os
import time
import pandas as pd
import lingam
import numpy as np
import pydot
import causal_orderings
from lingam.utils import make_dot
from typing import Dict, List, Optional, Set, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import itertools

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

def row_to_timeseries(row: pd.Series):
    """
    Converts a single row of a DataFrame into a time series DataFrame.
    
    Args:
        df (pd.Series): A single row from the original DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame with one column representing the time series.
    """

    return None

def save_causal_graph(causal_graph, labels: List[str], output_path: str):
    """
    Saves the causal graph as a .dot file, naming each node after its
    corresponding dataset column instead of a positional index.

    Args:
        causal_graph: The causal graph.
        labels (List[str]): List of variable names (dataset column names).
        output_path (str): The file path where the .dot file will be saved.
    """
    try:
        nodes = causal_graph.G.get_nodes()
        assert len(labels) == len(nodes)

        dot = pydot.Dot(graph_type="digraph", fontsize=18)
        dot.obj_dict["attributes"]["dpi"] = 200

        for label in labels:
            dot.add_node(pydot.Node(label))

        def get_arrow_type(endpoint):
            if endpoint == Endpoint.TAIL:
                return "none"
            elif endpoint == Endpoint.ARROW:
                return "normal"
            elif endpoint == Endpoint.CIRCLE:
                return "odot"
            else:
                raise NotImplementedError()

        for edge in causal_graph.G.get_graph_edges():
            node1_label = labels[nodes.index(edge.get_node1())]
            node2_label = labels[nodes.index(edge.get_node2())]
            dot.add_edge(
                pydot.Edge(
                    node1_label,
                    node2_label,
                    dir="both",
                    arrowtail=get_arrow_type(edge.get_endpoint1()),
                    arrowhead=get_arrow_type(edge.get_endpoint2()),
                )
            )

        dot.write(path=output_path)

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: to_pydot failed ({e}).")


def impute_and_encode_features(df: pd.DataFrame) -> pd.DataFrame:
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
        imputed_array, columns=feature_names
    )

    if categorical_cols:
        df_imputed = pd.get_dummies(
            df_imputed, columns=categorical_cols, drop_first=True
        )

    return df_imputed.apply(pd.to_numeric)


def generate_tiered_forbidden_edges(
    tiers: List[List[str]],
    forbid_within_tier: Optional[List[int]] = None,
    protected_tiers: Optional[List[int]] = None,
) -> Set[Tuple[str, str]]:
    """Generates forbidden edges based on a partial causal ordering.

    Rules:
      1. A variable in Tier j cannot cause a variable in Tier i if j > i (no backward causation).
      2. Optional: forbid edges between variables within specific tiers (e.g. randomized treatments).
      3. Optional: no variable can cause a variable that belongs to a protected tier.

    Args:
        tiers: The partial causal order.
        forbid_within_tier: Optional list of tier indices where causation within the
                            tier itself is also forbidden (e.g., [1] if Tier 1 is randomized).
        protected_tiers: Optional list of tier indices whose variables cannot be caused at all.

    Returns:
        Set of tuples (source, target) indicating that source -/-> target is forbidden.
    """
    forbidden_edges: Set[Tuple[str, str]] = set()
    forbid_within_tier = forbid_within_tier or []
    protected_tiers = protected_tiers or []

    num_tiers = len(tiers)

    # 1. Temporal rule: No backward causation (Tier j -/-> Tier i for all j > i)
    for j in range(num_tiers):
        for i in range(j):
            later_vars = tiers[j]
            earlier_vars = tiers[i]
            for source in later_vars:
                for target in earlier_vars:
                    forbidden_edges.add((source, target))

    # 2. Optional: Forbid internal causation within selected tiers
    for tier_idx in forbid_within_tier:
        if 0 <= tier_idx < num_tiers:
            vars_in_tier = tiers[tier_idx]
            for src in vars_in_tier:
                for dst in vars_in_tier:
                    if src != dst:
                        forbidden_edges.add((src, dst))

    # 3. Optional: Protected tiers cannot be caused at all
    for tier_idx in protected_tiers:
        for i in range(tier_idx):
            earlier_vars = tiers[i]
            for source in earlier_vars:
                for target in tiers[tier_idx]:
                    forbidden_edges.add((source, target))

    return forbidden_edges


def export_to_causallearn_bk(
    all_vars: List[str],
    forbidden: Set[Tuple[str, str]],
    required: Optional[Set[Tuple[str, str]]] = None,
):
    """Integrates forbidden (and optional required) constraints into causal-learn's

    BackgroundKnowledge object.
    """
    try:

        nodes: Dict[str, GraphNode] = {var: GraphNode(var) for var in all_vars}
        bk = BackgroundKnowledge()

        for src, dst in forbidden:
            bk.add_forbidden_by_node(nodes[src], nodes[dst])

        if required:
            for src, dst in required:
                bk.add_required_by_node(nodes[src], nodes[dst])

        return bk, nodes
    except ImportError:
        print("causal-learn is not installed in the local environment.")
        return None, None


def generate_background_knowledge(df, causal_orders, internally_forbidden_tiers = None, protected_tiers = None, required_edges=None):
    # Generate forbidden edges
    forbidden = generate_tiered_forbidden_edges(causal_orders, internally_forbidden_tiers, protected_tiers)

    # Flatten variable list to inspect coverage
    all_vars = df.columns

    return export_to_causallearn_bk(all_vars, forbidden, required_edges)


def perform_causal_discovery(
    df: pd.DataFrame, df_name: str, background_knowledge: Optional[BackgroundKnowledge] = None
) -> Tuple[Optional[object], float]:
    """Runs PC causal discovery and returns the fitted model and elapsed time."""
    print(
        f"[{time.strftime('%H:%M:%S')}] INFO: Running PC causal discovery..."
    )
    start_time = time.time()

    try:
        df_numpy = df.to_numpy().astype(float)

        causal_graph = pc(data=df_numpy, background_knowledge=background_knowledge)
        elapsed_time = time.time() - start_time
        print(
            f"[{time.strftime('%H:%M:%S')}] INFO: Causal discovery completed in {elapsed_time:.2f} seconds."
        )
        return causal_graph, elapsed_time
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S')}] ERROR: PC fitting failed for {df_name}. Reason: {e}"
        )
        return None, 0.0


def process_single_dataset(df: pd.DataFrame, df_name: str, background_knowledge: Optional[BackgroundKnowledge] = None) -> None:
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
    df_processed = impute_and_encode_features(df)
    if df_processed.empty or df_processed.shape[1] < 2:
        print(
            f"[{time.strftime('%H:%M:%S')}] WARNING: Dataset '{df_name}' has insufficient numeric data after preprocessing. Skipping."
        )
        return
    

    # 2. Causal Discovery
    model, _ = perform_causal_discovery(df_processed, df_name, background_knowledge=background_knowledge)
    if model is None:
        return

    # 3. Export
    labels = df_processed.columns.tolist()
    output_dot_path = f"causal_graphs/static_{df_name}_causal_graph.dot"
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

    print("==================================================")
    print(f"Starting Causal Discovery Pipeline for {len(datasets)} dataset(s).")
    print("==================================================\n")
    
    for index, row in datasets.iterrows():
        if pd.notna(row['file_path']):
            df = pd.read_csv(row['file_path'])
            background_knowledge = generate_background_knowledge(df, 
                                                                 row['causal_orders'], 
                                                                 row['internally_forbidden_tiers'], 
                                                                 row['protected_tiers'],
                                                                 row['required_edges']
                                                                )
            process_single_dataset(df, f"trial{index}")

    print("==================================================")
    print("Pipeline execution completed.")
    print("==================================================")

if __name__ == "__main__":

    dataset_paths = {
            2:   "cleaned_data/trial2.csv",
            6:   "cleaned_data/trial6.csv",
            13:  "cleaned_data/trial13.csv",
            29:  "cleaned_data/trial29.csv",
            37:  "cleaned_data/trial37.csv",
            108: "cleaned_data/trial108.csv",
            116: "cleaned_data/trial116.csv",
            120: "cleaned_data/trial120.csv",
    }

    datasets = causal_orderings.df_background_knowledge.copy()

    datasets = datasets.drop(29)
    datasets['file_path'] = dataset_paths

    run_pipeline(datasets)
