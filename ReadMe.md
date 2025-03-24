# Token Visualization API

This project provides a FastAPI interface for the Inseq feature attribution library, allowing you to run attribution jobs asynchronously and visualize token importance in language model outputs.

## Features

- Submit attribution jobs with configurable parameters
- Check job status and retrieve results
- Visualize attribution scores and step scores 
- Compare different attribution methods

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/token_visualization.git
   cd token_visualization
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```
   python api/main.py
   ```

The API will be available at http://localhost:8000.

## Quick Test

To verify that everything is working correctly, run:

```
python test_api.py
```

This will run an end-to-end test of the API with a small model (distilgpt2) and generate sample output.

## API Endpoints

- `POST /attribution/jobs`: Submit a new attribution job
- `GET /attribution/jobs/{job_id}`: Get job status
- `GET /attribution/jobs/{job_id}/results`: Get job results
- `GET /attribution/jobs`: List all jobs

## Usage

### Using the API Client

We provide a simple client for interacting with the API. Here's how to use it:

```python
from api_client import submit_attribution_job, wait_for_job_completion, get_job_results, visualize_attributions, create_attribution_dataframe

# Submit a job with a small model
job_id = submit_attribution_job(
    input_text="The quick brown fox jumps over the lazy dog.",
    model="distilgpt2",  # Small model that works well on CPU
    method="input_x_gradient",  # Fast attribution method
    step_scores=["probability"],
    force_cpu=True
)

# Wait for the job to complete
wait_for_job_completion(job_id)

# Get the results
results = get_job_results(job_id)

# Create a DataFrame and visualize the results
df = create_attribution_dataframe(results)
visualize_attributions(df)
```

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
           "step_scores": ["probability"],
           "force_cpu": True,
           "n_steps": 10
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

### Using the CLI Test Script

For quick testing, you can use the provided test script:

```bash
# Run the end-to-end test
python test_api.py

# Or test specific attribution in the API
python api/test_attribution.py
```

### Using the Jupyter Notebook

We provide a Jupyter notebook (`main.ipynb`) with helper functions for interacting with the API and visualizing the results. See the notebook for detailed examples.

## Project Structure

- `/api`: API implementation
  - `main.py`: FastAPI application and endpoints
  - `models.py`: Pydantic models for request and response data
  - `job_manager.py`: Job management and execution
  - `test_attribution.py`: Test script for the API
- `api_client.py`: Python client for the API
- `test_api.py`: End-to-end test script
- `main.ipynb`: Jupyter notebook for visualization
- `/attribution_output`: Output directory for attribution results

## Attribution Methods

The API supports the most reliable attribution methods from the Inseq library:

- `input_x_gradient`: Input multiplied by gradient (fast, recommended for testing)
- `saliency`: Simple gradient (fast)
- `integrated_gradients`: Integrated gradients (higher quality but slower)
- `attention`: Attention weights (fast but less reliable)

## Recommended Models for Testing

For the best experience during development and testing, we recommend using:

- `distilgpt2` (default): Very small model (82M parameters), works well on CPU
- `gpt2`: Small model (124M parameters), better quality but slower
- `facebook/opt-125m`: Good balance of size and quality

## Examples

Check the Jupyter notebook and API client for complete examples of:

1. Running attribution jobs
2. Creating attribution heatmaps
3. Plotting step scores
4. Finding important tokens
5. Comparing different attribution methods

## Data Format

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

This format matches the Inseq `FeatureAttributionOutput` data class structure, making it compatible with Inseq visualization tools.