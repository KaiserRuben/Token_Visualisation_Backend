# API

This project provides a FastAPI interface for the Inseq feature attribution library, allowing you to run attribution jobs asynchronously and visualize the results.

## Features

- Submit attribution jobs with configurable parameters
- Check job status and retrieve results
- Visualize attribution scores and step scores
- Compare different attribution methods

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/inseq-feature-attribution-api.git
   cd inseq-feature-attribution-api
   ```

2. Install the required packages:
   ```
   pip install fastapi uvicorn inseq torch pandas numpy matplotlib seaborn networkx
   ```

3. Start the API server:
   ```
   python main.py
   ```

The API will be available at http://localhost:8000.

## API Endpoints

- `POST /attribution/jobs`: Submit a new attribution job
- `GET /attribution/jobs/{job_id}`: Get job status
- `GET /attribution/jobs/{job_id}/results`: Get job results
- `GET /attribution/jobs`: List all jobs

## Usage

### Using the API Directly

1. Submit a job:
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/attribution/jobs",
       json={
           "model": "distilgpt2",
           "method": "input_x_gradient",
           "input_text": "Feature attribution helps explain model predictions.",
           "step_scores": ["probability", "logit"],
           "force_cpu": True
       }
   )
   
   job_id = response.json()["job_id"]
   print(f"Job ID: {job_id}")
   ```

2. Check job status:
   ```python
   response = requests.get(f"http://localhost:8000/attribution/jobs/{job_id}")
   status = response.json()["status"]
   print(f"Job status: {status}")
   ```

3. Get job results when completed:
   ```python
   if status == "completed":
       results = requests.get(f"http://localhost:8000/attribution/jobs/{job_id}/results").json()
       print(results)
   ```

### Using the Jupyter Notebook

We provide a Jupyter notebook (`inseq_api_visualization.ipynb`) with helper functions for interacting with the API and visualizing the results. See the notebook for detailed examples.

## Project Structure

- `main.py`: FastAPI application and endpoints
- `models.py`: Pydantic models for request and response data
- `job_manager.py`: Job management and execution
- `inseq_api_visualization.ipynb`: Jupyter notebook for visualization

## Attribution Methods

The API supports all attribution methods available in the Inseq library:

- `input_x_gradient`: Input multiplied by gradient
- `saliency`: Simple gradient
- `integrated_gradients`: Integrated gradients
- `deep_lift`: DeepLIFT
- `lime`: Local Interpretable Model-agnostic Explanations
- `attention`: Attention weights
- And more...

## Examples

Check the Jupyter notebook for complete examples of:

1. Running attribution jobs
2. Creating attribution heatmaps
3. Plotting step scores
4. Finding important tokens
5. Comparing different attribution methods

## Notes on Data Format

The API returns attribution results in a format that can be easily used for visualization:

```json
{
  "sequence_attributions": [
    {
      "source": [{"token": "...", "token_id": 123}, ...],
      "target": [{"token": "...", "token_id": 456}, ...],
      "source_attributions": [[0.1, 0.2, ...], ...],
      "step_scores": {"probability": [0.8, 0.7, ...], ...}
    }
  ],
  "info": {...}
}
```

This format closely matches the Inseq `FeatureAttributionOutput` data class structure, making it compatible with Inseq visualization tools.