"""
Inseq Feature Attribution API Client

This module provides helper functions for interacting with the Inseq Feature Attribution API
and visualizing the results.
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Optional, Union, Any

# Default API base URL - modify this as needed
API_BASE_URL = "http://localhost:8000"


def submit_attribution_job(
    input_text: str,
    model: str = "distilgpt2",
    method: str = "input_x_gradient",
    step_scores: List[str] = ["probability"],
    force_cpu: bool = False,
    n_steps: int = 20,
    api_base_url: str = API_BASE_URL
) -> str:
    """
    Submit an attribution job to the API and return the job ID.

    Parameters:
    -----------
    input_text : str
        The input text to attribute
    model : str, optional
        The model to use for attribution (HuggingFace model name or path)
    method : str, optional
        The attribution method to use. Available options:
        - input_x_gradient, saliency, integrated_gradients, deep_lift, gradient_shap
        - layer_integrated_gradients, layer_gradient_x_activation, layer_deep_lift
        - attention, occlusion, lime, value_zeroing, reagent
        - discretized_integrated_gradients, sequential_integrated_gradients
    step_scores : list of str, optional
        Step scores to compute alongside attribution. Available options:
        - logit, probability, entropy, crossentropy, perplexity
        - contrast_logits, contrast_prob, contrast_logits_diff, contrast_prob_diff
        - pcxmi, kl_divergence, in_context_pvi, mc_dropout_prob_avg, top_p_size
    force_cpu : bool, optional
        Force CPU usage for all operations
    n_steps : int, optional
        Number of steps for methods that require it (integrated_gradients, deep_lift, etc.)
    api_base_url : str, optional
        Base URL for the API

    Returns:
    --------
    str
        The job ID
    """
    url = f"{api_base_url}/attribution/jobs"
    data = {
        "model": model,
        "method": method,
        "input_text": input_text,
        "step_scores": step_scores,
        "force_cpu": force_cpu,
        "n_steps": n_steps
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

    return response.json()["job_id"]


def check_job_status(job_id: str, api_base_url: str = API_BASE_URL) -> Dict[str, Any]:
    """
    Check the status of an attribution job.

    Parameters:
    -----------
    job_id : str
        The job ID to check
    api_base_url : str, optional
        Base URL for the API

    Returns:
    --------
    dict
        Job status information
    """
    url = f"{api_base_url}/attribution/jobs/{job_id}"
    response = requests.get(url)
    response.raise_for_status()

    return response.json()


def get_job_results(job_id: str, api_base_url: str = API_BASE_URL, max_retries: int = 3, retry_interval: int = 2) -> Dict[str, Any]:
    """
    Get the results of a completed attribution job.

    Parameters:
    -----------
    job_id : str
        The job ID to retrieve results for
    api_base_url : str, optional
        Base URL for the API
    max_retries : int, optional
        Maximum number of retries if the request fails
    retry_interval : int, optional
        Seconds to wait between retries

    Returns:
    --------
    dict
        Attribution results

    Raises:
    -------
    Exception
        If the request fails after all retries
    """
    url = f"{api_base_url}/attribution/jobs/{job_id}/results"

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error retrieving results (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
            else:
                raise Exception(f"Failed to retrieve results after {max_retries} attempts: {e}")



def wait_for_job_completion(
    job_id: str,
    poll_interval: int = 2,
    max_attempts: int = 60,
    api_base_url: str = API_BASE_URL
) -> Dict[str, Any]:
    """
    Wait for a job to complete, polling at regular intervals.

    Parameters:
    -----------
    job_id : str
        The job ID to wait for
    poll_interval : int, optional
        Seconds to wait between status checks
    max_attempts : int, optional
        Maximum number of status checks before timing out
    api_base_url : str, optional
        Base URL for the API

    Returns:
    --------
    dict
        Final job status

    Raises:
    -------
    Exception
        If the job fails
    TimeoutError
        If the job doesn't complete within the allotted time
    """
    attempts = 0
    while attempts < max_attempts:
        job_status = check_job_status(job_id, api_base_url)

        if job_status["status"] == "completed":
            return job_status
        elif job_status["status"] == "failed":
            raise Exception(f"Job failed: {job_status.get('error', 'Unknown error')}")

        print(f"Job status: {job_status['status']}. Waiting {poll_interval} seconds...")
        time.sleep(poll_interval)
        attempts += 1

    raise TimeoutError(f"Job did not complete after {max_attempts * poll_interval} seconds")


def create_attribution_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame from the attribution results.

    Parameters:
    -----------
    results : dict
        Attribution results from the API

    Returns:
    --------
    pandas.DataFrame
        DataFrame with source tokens as indices, target tokens as columns,
        and attribution scores as values
    """
    sequence_attr = results["sequence_attributions"][0]

    source_tokens = [token["token"] for token in sequence_attr["source"]]
    target_tokens = [token["token"] for token in sequence_attr["target"]]

    attribution_scores = np.array(sequence_attr["source_attributions"])

    df = pd.DataFrame(attribution_scores, index=source_tokens, columns=target_tokens)
    return df


def visualize_attributions(df: pd.DataFrame, figsize: tuple = (14, 10)) -> None:
    """
    Create a heatmap visualization of attribution scores.

    Parameters:
    -----------
    df : pandas.DataFrame
        Attribution scores DataFrame (from create_attribution_dataframe)
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                linewidths=.5, xticklabels=True, yticklabels=True)
    plt.title("Attribution Scores: Source Tokens (y) → Target Tokens (x)")
    plt.tight_layout()
    plt.show()


def plot_step_scores(results: Dict[str, Any], figsize: tuple = (12, 6)) -> Optional[pd.DataFrame]:
    """
    Plot step scores for each token.

    Parameters:
    -----------
    results : dict
        Attribution results from the API
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with step scores, or None if no step scores are available
    """
    sequence_attr = results["sequence_attributions"][0]

    if not sequence_attr.get("step_scores"):
        print("No step scores available in the results.")
        return None

    target_tokens = [token["token"] for token in sequence_attr["target"]]

    # Create a dataframe for step scores
    step_scores_data = {}
    for score_name, scores in sequence_attr["step_scores"].items():
        step_scores_data[score_name] = scores

    df_scores = pd.DataFrame(step_scores_data, index=target_tokens)

    # Plot
    plt.figure(figsize=figsize)
    for column in df_scores.columns:
        plt.plot(df_scores.index, df_scores[column], marker='o', label=column)

    plt.xlabel("Target Tokens")
    plt.ylabel("Score")
    plt.title("Step Scores")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return df_scores


def find_important_tokens(df_attribution: pd.DataFrame) -> pd.DataFrame:
    """
    Find the most important source token for each target token.

    Parameters:
    -----------
    df_attribution : pandas.DataFrame
        Attribution scores DataFrame (from create_attribution_dataframe)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with target tokens, their most important source tokens, and scores
    """
    important_tokens = []

    for target_token in df_attribution.columns:
        most_important_source = df_attribution[target_token].abs().idxmax()  # Using absolute value
        score = df_attribution[target_token][most_important_source]

        important_tokens.append({
            "target_token": target_token,
            "source_token": most_important_source,
            "score": score
        })

    return pd.DataFrame(important_tokens)


def plot_multiple_step_scores(df_scores: pd.DataFrame) -> None:
    """
    Create separate plots for each step score.

    Parameters:
    -----------
    df_scores : pandas.DataFrame
        DataFrame with step scores (from plot_step_scores)
    """
    fig, axes = plt.subplots(
        len(df_scores.columns), 1,
        figsize=(12, 4*len(df_scores.columns)),
        sharex=True
    )

    # Handle case with only one score
    if len(df_scores.columns) == 1:
        axes = [axes]

    for i, column in enumerate(df_scores.columns):
        axes[i].plot(df_scores.index, df_scores[column], marker='o', color=f'C{i}')
        axes[i].set_title(f"{column} Step Score")
        axes[i].set_ylabel("Score")
        axes[i].grid(True, linestyle='--', alpha=0.7)

    axes[-1].set_xlabel("Target Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_attribution_flow(df_attribution: pd.DataFrame, threshold: float = 0.1) -> Any:
    """
    Create a visualization of attribution flow from source to target tokens.

    Parameters:
    -----------
    df_attribution : pandas.DataFrame
        Attribution scores DataFrame (from create_attribution_dataframe)
    threshold : float, optional
        Minimum absolute score value to include in the visualization

    Returns:
    --------
    networkx.DiGraph
        The directed graph object used for visualization
    """
    import networkx as nx

    # Create a graph
    G = nx.DiGraph()

    # Add source and target nodes
    source_tokens = df_attribution.index
    target_tokens = df_attribution.columns

    # Add nodes
    for i, token in enumerate(source_tokens):
        G.add_node(f"s_{i}", label=token, type="source")

    for i, token in enumerate(target_tokens):
        G.add_node(f"t_{i}", label=token, type="target")

    # Add edges with weights above threshold
    for i, s_token in enumerate(source_tokens):
        for j, t_token in enumerate(target_tokens):
            score = df_attribution.loc[s_token, t_token]
            if abs(score) > threshold:
                G.add_edge(f"s_{i}", f"t_{j}", weight=score)

    # Plot
    plt.figure(figsize=(12, 8))

    # Define positions
    pos = {}
    for i, token in enumerate(source_tokens):
        pos[f"s_{i}"] = (0, -i)

    for i, token in enumerate(target_tokens):
        pos[f"t_{i}"] = (1, -i)

    # Draw nodes
    source_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "source"]
    target_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "target"]

    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color="lightblue", node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color="lightgreen", node_size=500)

    # Draw edges with colors based on weight
    edges = G.edges(data=True)
    positive_edges = [(u, v) for u, v, d in edges if d["weight"] > 0]
    negative_edges = [(u, v) for u, v, d in edges if d["weight"] < 0]

    # Draw edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color="green", width=2, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color="red", width=2, alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Draw labels
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("Attribution Flow: Source Tokens → Target Tokens")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G


def run_attribution_workflow(
    input_text: str,
    model: str = "distilgpt2",
    method: str = "input_x_gradient",
    step_scores: List[str] = ["probability", "logit"],
    force_cpu: bool = True,
    api_base_url: str = API_BASE_URL
) -> Dict[str, Any]:
    """
    Run a complete attribution workflow from submission to visualization.

    Parameters:
    -----------
    input_text : str
        The input text to attribute
    model : str, optional
        The model to use for attribution
    method : str, optional
        The attribution method to use
    step_scores : List[str], optional
        Step scores to compute
    force_cpu : bool, optional
        Whether to force CPU usage
    api_base_url : str, optional
        Base URL for the API

    Returns:
    --------
    dict
        Dictionary containing job info, dataframes and results
    """
    # 1. Submit job
    print(f"Submitting attribution job for text: '{input_text}'")
    job_id = submit_attribution_job(
        input_text=input_text,
        model=model,
        method=method,
        step_scores=step_scores,
        force_cpu=force_cpu,
        api_base_url=api_base_url
    )
    print(f"Job submitted with ID: {job_id}")

    # 2. Wait for job completion
    print("Waiting for job to complete...")
    job_status = wait_for_job_completion(job_id, api_base_url=api_base_url)
    print(f"Job completed with status: {job_status['status']}")

    # 3. Get results
    print("Retrieving results...")
    results = get_job_results(job_id, api_base_url=api_base_url)

    # 4. Create attribution dataframe
    df_attribution = create_attribution_dataframe(results)
    print("\nAttribution Scores DataFrame:")
    print(df_attribution)

    # 5. Visualize attribution scores
    print("\nVisualizing attribution scores...")
    visualize_attributions(df_attribution)

    # 6. Plot step scores
    print("\nPlotting step scores...")
    df_step_scores = plot_step_scores(results)

    # 7. Find important tokens
    print("\nFinding important tokens...")
    df_important = find_important_tokens(df_attribution)
    print("\nMost important source token for each target token:")
    print(df_important)

    # 8. Print attribution info
    print("\nAttribution Information:")
    print(json.dumps(results["info"], indent=2))

    return {
        "job_id": job_id,
        "df_attribution": df_attribution,
        "df_step_scores": df_step_scores,
        "df_important": df_important,
        "results": results
    }


def compare_attribution_methods(
    input_text: str,
    methods: List[str] = ["input_x_gradient", "saliency", "integrated_gradients", "attention"],
    model: str = "distilgpt2",
    api_base_url: str = API_BASE_URL
) -> Dict[str, Any]:
    """
    Run attribution with multiple methods and compare the results.

    This function submits multiple attribution jobs with different methods and
    creates visualizations to compare the results. It's useful for understanding
    how different attribution methods work on the same input.

    Parameters:
    -----------
    input_text : str
        The input text to attribute
    methods : list of str
        List of attribution methods to compare
    model : str
        The model to use for attribution
    api_base_url : str, optional
        Base URL for the API

    Returns:
    --------
    dict
        Dictionary containing all results and dataframes for each method
    """
    results = {}
    dataframes = {}

    for method in methods:
        print(f"\nRunning attribution with method: {method}")
        job_id = submit_attribution_job(
            input_text=input_text,
            model=model,
            method=method,
            step_scores=["probability"],
            force_cpu=True,
            api_base_url=api_base_url
        )

        print(f"Job submitted with ID: {job_id}")
        job_status = wait_for_job_completion(job_id, api_base_url=api_base_url)
        print(f"Job completed with status: {job_status['status']}")

        result = get_job_results(job_id, api_base_url=api_base_url)
        results[method] = result

        df = create_attribution_dataframe(result)
        dataframes[method] = df

        # Visualize this method's results
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                    linewidths=.5, xticklabels=True, yticklabels=True)
        plt.title(f"Attribution Scores with {method}")
        plt.tight_layout()
        plt.show()

    return {
        "results": results,
        "dataframes": dataframes
    }


# Method summary information
ATTRIBUTION_METHODS = {
    "input_x_gradient": {
        "description": "Input multiplied by gradient",
        "complexity": "Low",
        "speed": "Fast",
        "special_params": None
    },
    "saliency": {
        "description": "Simple gradient attribution",
        "complexity": "Low",
        "speed": "Fast",
        "special_params": None
    },
    "integrated_gradients": {
        "description": "Integrated Gradients method",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "deep_lift": {
        "description": "DeepLIFT method",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "gradient_shap": {
        "description": "GradientSHAP method",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "layer_integrated_gradients": {
        "description": "Layer Integrated Gradients",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "layer_gradient_x_activation": {
        "description": "Layer Gradient × Activation",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": None
    },
    "layer_deep_lift": {
        "description": "Layer DeepLIFT",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "attention": {
        "description": "Attention weights",
        "complexity": "Low",
        "speed": "Fast",
        "special_params": None
    },
    "occlusion": {
        "description": "Occlusion-based attribution",
        "complexity": "Medium",
        "speed": "Slow",
        "special_params": None
    },
    "lime": {
        "description": "LIME-based attribution",
        "complexity": "High",
        "speed": "Slow",
        "special_params": None
    },
    "value_zeroing": {
        "description": "Value Zeroing method",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": None
    },
    "reagent": {
        "description": "Recursive attribution generator",
        "complexity": "High",
        "speed": "Slow",
        "special_params": None
    },
    "discretized_integrated_gradients": {
        "description": "Discretized IG",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    },
    "sequential_integrated_gradients": {
        "description": "Sequential IG",
        "complexity": "Medium",
        "speed": "Medium",
        "special_params": "n_steps"
    }
}

STEP_SCORE_FUNCTIONS = {
    "logit": "Logit of the target token",
    "probability": "Probability of the target token",
    "entropy": "Entropy of output distribution",
    "crossentropy": "Cross entropy between target and logits",
    "perplexity": "Perplexity of target from logits",
    "contrast_logits": "Logit of a generation given contrastive context",
    "contrast_prob": "Probability given contrastive context",
    "contrast_logits_diff": "Difference between logits with contrastive inputs",
    "contrast_prob_diff": "Difference between probabilities with contrastive inputs",
    "pcxmi": "Pointwise conditional cross-mutual information",
    "kl_divergence": "KL divergence between distributions",
    "in_context_pvi": "In-context pointwise V-usable information",
    "mc_dropout_prob_avg": "Monte Carlo dropout probability average",
    "top_p_size": "Number of tokens with cumulative probability above threshold"
}

# Model recommendations
RECOMMENDED_MODELS = {
    "distilgpt2": "Lightweight GPT-2 model, works well on CPU",
    "gpt2": "Standard GPT-2 model",
    "facebook/opt-125m": "Small OPT model",
    "google/flan-t5-small": "Small T5 model (encoder-decoder)"
}