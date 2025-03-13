import inseq
import torch
import argparse
import os
from pathlib import Path


def patch_tokenizer(model):
    """Add unk_token_id to tokenizer if it doesn't exist."""
    if model.tokenizer.unk_token_id is None:
        print("Setting default unk_token_id to 0")
        model.tokenizer.unk_token = model.tokenizer.convert_ids_to_tokens(0)
        model.tokenizer.unk_token_id = 0
    return model


def determine_device():
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device, device  # For CUDA, use same device for model and attribution
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("MPS detected for model loading, but attribution will use CPU")
        return device, 'cpu'  # For MPS, use CPU for attribution
    else:
        device = 'cpu'
        print("Using CPU")
        return device, device  # For CPU, use CPU for both


def validate_method(method, available_methods):
    """Validate if the requested attribution method is available."""
    if method not in available_methods:
        print(f"Warning: Method '{method}' not available.")
        print(f"Available methods: {', '.join(available_methods)}")
        print(f"Falling back to method: {available_methods[0]}")
        return available_methods[0]
    return method


def save_attribution_results(attribution, output_dir):
    """Save attribution results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Define the file path
        json_path = output_dir / 'attribution.json'

        # Use Inseq's native save method with minimal parameters for compatibility
        attribution.save(
            path=str(json_path),
            overwrite=True  # Important to avoid errors with existing files
        )

        print(f"Attribution saved to {json_path}")

        # Verify the file was created
        if json_path.exists():
            file_size = json_path.stat().st_size
            print(f"File size: {file_size / 1024:.2f} KB")
            return True
        else:
            print("Warning: File was not created despite no errors")
            return False

    except Exception as e:
        print(f"Error saving attribution results: {e}")
        print(f"Trying with minimal parameters...")

        try:
            # Fallback to absolute minimal parameters
            attribution.save(str(json_path))
            print(f"Attribution saved to {json_path} with minimal parameters")
            return True
        except Exception as e:
            print(f"Still failed: {e}")
            return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inseq Feature Attribution')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model to use for attribution')
    parser.add_argument('--method', type=str, default='input_x_gradient',
                        help='Feature attribution method to use')
    parser.add_argument('--input', type=str,
                        default='Write me a short poem about programming.',
                        help='Input text for attribution')
    parser.add_argument('--n_steps', type=int, default=20,
                        help='Number of steps for methods that require it (only applies to certain methods)')
    parser.add_argument('--output_dir', type=str, default='./attribution_output',
                        help='Directory to save output files')
    parser.add_argument('--step_scores', nargs='+', default=['probability'],
                        help='Step scores to compute alongside attribution')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage for all operations')
    args = parser.parse_args()

    # Determine devices for model loading and attribution
    model_device, attr_device = ('cpu', 'cpu') if args.force_cpu else determine_device()

    # Validate attribution method
    available_methods = inseq.list_feature_attribution_methods()
    method = validate_method(args.method, available_methods)

    # For MPS compatibility, suggest simpler methods
    if model_device == 'mps' and method in ['integrated_gradients', 'deep_lift', 'gradient_shap']:
        print(f"Note: {method} might have compatibility issues with MPS.")
        print("Consider using 'input_x_gradient' or 'saliency' for better compatibility.")

    # Load model with attribution method
    print(f"Loading model {args.model} with attribution method {method}...")
    try:
        # First load the model on the detected device
        model = inseq.load_model(args.model, method)

        # Apply patch for unk_token_id
        model = patch_tokenizer(model)

        # Move model to device
        if hasattr(model, 'to'):
            try:
                model.to(model_device)
                print(f"Model moved to {model_device}")
            except Exception as e:
                print(f"Could not move model to {model_device}: {e}")
                print("Continuing with default device")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Perform attribution with appropriate parameters based on the method
    print(f"Performing attribution on input: {args.input}")

    try:
        # Move model to attribution device if different from model device
        if attr_device != model_device and hasattr(model, 'to'):
            model.to(attr_device)
            print(f"Model moved to {attr_device} for attribution")

        # Use method-specific parameters
        attribution_kwargs = {}

        # Add n_steps only for methods that require it
        if method in ['integrated_gradients', 'deep_lift', 'gradient_shap']:
            attribution_kwargs['n_steps'] = args.n_steps

        # Include step scores for all methods
        attribution_kwargs['step_scores'] = args.step_scores

        # Perform the attribution
        attribution = model.attribute(args.input, **attribution_kwargs)

        # Save the results
        if save_attribution_results(attribution, args.output_dir):
            print("Attribution results saved successfully")
        else:
            print("Failed to save attribution results")

        # Display attribution in console
        print("\nAttribution results:")
        attribution.show()

    except Exception as e:
        print(f"Error during attribution: {e}")
        print("\nTry one of these alternatives:")
        print("1. Use a simpler attribution method like 'input_x_gradient' or 'saliency'")
        print("2. If using Apple Silicon (M1/M2/M3), use the --force_cpu flag")
        print("3. Try different attribution step scores (e.g., --step_scores probability logit)")


if __name__ == "__main__":
    main()