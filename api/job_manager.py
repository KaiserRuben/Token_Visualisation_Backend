import asyncio
import logging
import uuid
from pathlib import Path
import torch
import inseq
from models import JobStatus, AttributionJobRequest
import multiprocessing as mp
from functools import partial
import gc
import signal
import faulthandler
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    """
    Manager for attribution jobs.

    This class handles the creation, execution, and management of attribution jobs.
    """

    def __init__(self, timeout=1800):
        """
        Initialize the job manager.

        Args:
            timeout (int): Maximum time in seconds to wait for a job to complete.
                           Default is 1800 (30 minutes).
        """
        self.jobs = {}
        self.output_dir = Path("./attribution_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def create_job(self, request: AttributionJobRequest) -> str:
        """
        Create a new job.

        Args:
            request: The job request.

        Returns:
            str: The job ID.
        """
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "request": request.dict(),
            "status": JobStatus.PENDING,
            "output_file": None,
            "error": None
        }
        return job_id

    def get_job(self, job_id: str) -> dict:
        """
        Get a job by ID.

        Args:
            job_id: The job ID.

        Returns:
            dict: The job data.

        Raises:
            ValueError: If the job is not found.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def list_jobs(self) -> list:
        """
        List all jobs.

        Returns:
            list: A list of all jobs with their IDs and statuses.
        """
        return [
            {"job_id": job_id, "status": self.jobs[job_id]["status"]}
            for job_id in self.jobs
        ]

    async def run_job(self, job_id: str):
        """
        Run a job asynchronously.

        Args:
            job_id: The job ID.
        """
        # Get job request
        job = self.jobs[job_id]
        request = AttributionJobRequest(**job["request"])

        # Update job status
        job["status"] = JobStatus.RUNNING

        try:
            # Set up multiprocessing environment with explicit worker count
            # These need to be set BEFORE the process starts
            os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["NUMEXPR_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(mp.cpu_count())
            os.environ["OPENBLAS_NUM_THREADS"] = str(mp.cpu_count())

            # Explicitly tell PyTorch to use parallel backends
            # This helps ensure the parallel backend is activated
            os.environ["OMP_PLACES"] = "cores"
            os.environ["OMP_PROC_BIND"] = "close"

            # Use 'spawn' method for better compatibility with PyTorch
            # (only set if not already configured)
            mp_ctx = mp.get_context('spawn')

            # Create a Queue to get the result
            result_queue = mp_ctx.Queue()

            # Create a function that will run in a separate process
            execute_fn = partial(
                self._execute_attribution_safe,
                job_id=job_id,
                request=request,
                result_queue=result_queue
            )

            # Start the process
            logger.info(f"Starting attribution process for job {job_id}")
            process = mp_ctx.Process(target=execute_fn)
            process.start()

            # Wait for the process to finish with a much longer timeout
            process.join(timeout=self.timeout)  # Using the configurable timeout

            # Check if the process is still alive
            if process.is_alive():
                logger.error(f"Attribution process for job {job_id} timed out")
                process.terminate()
                process.join()
                job["status"] = JobStatus.FAILED
                job["error"] = f"Attribution process timed out after {self.timeout} seconds"
                return

            # Check if the process exited with an error
            if process.exitcode != 0:
                logger.error(f"Attribution process for job {job_id} failed with exit code {process.exitcode}")
                job["status"] = JobStatus.FAILED
                job["error"] = f"Attribution process failed with exit code {process.exitcode}"
                return

            # Get the result from the queue
            if result_queue.empty():
                logger.error(f"No result returned from attribution process for job {job_id}")
                job["status"] = JobStatus.FAILED
                job["error"] = "No result returned from attribution process"
                return

            result = result_queue.get()

            # Check if the result is an error
            if isinstance(result, Exception):
                logger.error(f"Error in attribution process for job {job_id}: {result}")
                job["status"] = JobStatus.FAILED
                job["error"] = str(result)
                return

            # Update job with result
            job["status"] = JobStatus.COMPLETED
            job["output_file"] = result

        except Exception as e:
            logger.exception(f"Error running job {job_id}")
            job["status"] = JobStatus.FAILED
            job["error"] = str(e)

    def _execute_attribution_safe(self, job_id: str, request: AttributionJobRequest, result_queue):
        """
        Execute the attribution job in a safe manner that won't crash the main process.
        This method is designed to be run in a separate process.

        Args:
            job_id: The job ID.
            request: The job request.
            result_queue: Queue to put the result or exception
        """
        try:
            # Import modules inside the process - this is necessary when using spawn
            import os
            import torch
            import gc
            import signal
            import faulthandler
            import multiprocessing as mp

            # Configure process isolation and threading
            # These need to be set INSIDE the process too
            os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["NUMEXPR_NUM_THREADS"] = str(mp.cpu_count())
            os.environ["OPENBLAS_NUM_THREADS"] = str(mp.cpu_count())

            # Configure PyTorch to use multiple cores
            torch.set_num_threads(mp.cpu_count())

            # Configure both intra-op and inter-op parallelism
            # This is crucial for proper core utilization
            if hasattr(torch, 'set_num_interop_threads'):
                # For PyTorch 1.5+
                torch.set_num_interop_threads(min(mp.cpu_count(), 4))

            # Enable parallel backends for operations
            try:
                import torch.backends.mkldnn
                torch.backends.mkldnn.enabled = True
            except:
                pass

            # Enable faulthandler to get better crash reports
            faulthandler.enable()

            # Define signal handler for SIGBUS (Bus error)
            def handle_bus_error(sig, frame):
                error_msg = "Memory alignment error (SIGBUS) occurred during attribution"
                logger.error(error_msg)
                result_queue.put(Exception(error_msg))
                # Exit process with error
                import os
                os._exit(1)

            # Define signal handler for SIGSEGV (Segmentation fault)
            def handle_segfault(sig, frame):
                error_msg = "Memory access error (SIGSEGV) occurred during attribution"
                logger.error(error_msg)
                result_queue.put(Exception(error_msg))
                # Exit process with error
                import os
                os._exit(1)

            # Register signal handlers
            signal.signal(signal.SIGBUS, handle_bus_error)
            signal.signal(signal.SIGSEGV, handle_segfault)

            # Execute the attribution with memory limits and careful error handling
            from models import AttributionJobRequest  # Import here for spawn context
            import inseq  # Import inseq inside process

            # Execute attribution directly in this process to avoid nested import issues
            try:
                # Determine devices
                model_device, attr_device = self._determine_device(request.force_cpu)

                # Validate attribution method
                available_methods = inseq.list_feature_attribution_methods()
                method = self._validate_method(request.method, available_methods)

                logger.info(f"Loading model {request.model} with method {method}")

                # Rest of execution logic from _execute_attribution...
                # Create a memory-efficient environment
                gc.collect()

                # Log configuration
                logger.info(f"PyTorch is using {torch.get_num_threads()} threads")
                logger.info(f"CPU count: {mp.cpu_count()}")

                # Warm up cores
                def warm_up_cores(device):
                    logger.info("Warming up CPU cores before attribution...")
                    a = torch.randn(1000, 1000, device=device)
                    b = torch.randn(1000, 1000, device=device)
                    for _ in range(5):
                        _ = torch.mm(a, b)
                        _ = torch.nn.functional.relu(a)
                    logger.info("Core warm-up complete")

                warm_up_cores(attr_device)

                # Continue with model loading and attribution...
                # (Using the same code as in _execute_attribution)

                # Prepare model loading parameters
                model_kwargs = {"attn_implementation": "eager"}
                tokenizer_kwargs = {
                    "padding_side": "left",
                    "pad_token": "<pad>"
                }

                # Load model
                try:
                    model = inseq.load_model(
                        request.model,
                        method,
                        model_kwargs=model_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs
                    )
                    logger.info(f"Model loaded successfully using standard approach")
                except Exception as first_error:
                    error_msg = str(first_error)
                    logger.warning(f"Error in first loading attempt: {error_msg}")

                    try:
                        if "does not appear to have a file named pytorch_model.bin but there is a file for Flax weights" in error_msg:
                            model_kwargs["from_flax"] = True
                            model = inseq.load_model(
                                request.model,
                                method,
                                model_kwargs=model_kwargs,
                                tokenizer_kwargs=tokenizer_kwargs
                            )
                            logger.info(f"Model loaded successfully using from_flax=True")
                        elif "gpt2" in request.model.lower():
                            import transformers
                            config = transformers.AutoConfig.from_pretrained(request.model)
                            tokenizer = transformers.AutoTokenizer.from_pretrained(request.model)
                            model_class = transformers.AutoModelForCausalLM.from_pretrained(request.model)
                            model = inseq.load_model(model_class, method, tokenizer=tokenizer)
                            logger.info(f"Model loaded successfully using transformers directly")
                        else:
                            raise first_error
                    except Exception as second_error:
                        logger.error(f"Failed all model loading attempts. Final error: {second_error}")
                        raise first_error

                # Patch tokenizer
                if model.tokenizer.unk_token_id is None:
                    model.tokenizer.unk_token = model.tokenizer.convert_ids_to_tokens(0)
                    model.tokenizer.unk_token_id = 0

                if model.tokenizer.pad_token is None:
                    if model.tokenizer.eos_token is not None:
                        logger.info(f"Setting pad_token to eos_token: {model.tokenizer.eos_token}")
                        model.tokenizer.pad_token = model.tokenizer.eos_token
                    else:
                        logger.info("Setting default pad_token to '<pad>'")
                        model.tokenizer.pad_token = "<pad>"

                # Move model to device
                if hasattr(model, 'to'):
                    try:
                        model.to(attr_device)
                        logger.info(f"Model moved to {attr_device}")
                    except Exception as e:
                        logger.warning(f"Could not move model to {attr_device}: {e}")

                # Perform attribution
                try:
                    if method in ['integrated_gradients', 'deep_lift', 'gradient_shap',
                                  'discretized_integrated_gradients', 'sequential_integrated_gradients',
                                  'layer_integrated_gradients', 'layer_deep_lift']:
                        logger.info(f"Using n_steps={request.n_steps} for {method}")
                        attribution = model.attribute(
                            request.input_text,
                            step_scores=request.step_scores,
                            n_steps=request.n_steps
                        )
                    else:
                        attribution = model.attribute(
                            request.input_text,
                            step_scores=request.step_scores
                        )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"CUDA out of memory error: {e}")
                        raise Exception(f"GPU out of memory. Try using force_cpu=True or a smaller model.")
                    else:
                        raise

                # Save results
                from pathlib import Path
                output_dir = Path("./attribution_output")
                output_dir.mkdir(parents=True, exist_ok=True)

                job_output_dir = output_dir / job_id
                job_output_dir.mkdir(parents=True, exist_ok=True)
                json_path = job_output_dir / 'attribution.json'

                attribution.save(str(json_path), overwrite=True, use_primitives=True)
                logger.info(f"Attribution results saved to {json_path}")

                result = str(json_path)
                result_queue.put(result)

            except Exception as attr_error:
                logger.error(f"Error during attribution: {attr_error}")
                result_queue.put(Exception(f"Attribution failed: {str(attr_error)}"))
        except Exception as e:
            # Put the exception in the queue
            result_queue.put(e)

    def _execute_attribution(self, job_id: str, request: AttributionJobRequest) -> str:
        """
        Execute the attribution job.

        Args:
            job_id: The job ID.
            request: The job request.

        Returns:
            str: The path to the output file.
        """
        try:
            # Log the PyTorch configuration for debugging
            logger.info(f"PyTorch is using {torch.get_num_threads()} threads")
            logger.info(f"CPU count: {mp.cpu_count()}")

            # Log thread/parallelism environment variables
            env_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND"]
            for var in env_vars:
                logger.info(f"{var}: {os.environ.get(var, 'not set')}")

            # Create a memory-efficient environment
            gc.collect()  # Force garbage collection before starting

            # Optional: Log CPU affinity (requires psutil)
            try:
                import psutil
                logger.info(f"Process CPU affinity: {psutil.Process().cpu_affinity()}")
            except (ImportError, AttributeError):
                pass

            # Determine devices
            model_device, attr_device = self._determine_device(request.force_cpu)

            # Validate attribution method
            available_methods = inseq.list_feature_attribution_methods()
            method = self._validate_method(request.method, available_methods)

            logger.info(f"Loading model {request.model} with method {method}")

            # Prepare model loading parameters - using minimal settings for stability
            model_kwargs = {
                "attn_implementation": "eager"  # Avoid warnings about scaled_dot_product_attention
            }
            tokenizer_kwargs = {
                "padding_side": "left",  # Some models need this for proper padding
                "pad_token": "<pad>"  # Ensure pad token is set
            }

            # Try various model loading approaches with appropriate error handling
            try:
                # First attempt - standard loading
                model = inseq.load_model(
                    request.model,
                    method,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
                logger.info(f"Model loaded successfully using standard approach")
            except Exception as first_error:
                error_msg = str(first_error)
                logger.warning(f"Error in first loading attempt: {error_msg}")

                try:
                    if "does not appear to have a file named pytorch_model.bin but there is a file for Flax weights" in error_msg:
                        # Second attempt - try with from_flax=True
                        logger.info(f"Detected Flax model, attempting to load with from_flax=True")
                        model_kwargs["from_flax"] = True
                        model = inseq.load_model(
                            request.model,
                            method,
                            model_kwargs=model_kwargs,
                            tokenizer_kwargs=tokenizer_kwargs
                        )
                        logger.info(f"Model loaded successfully using from_flax=True")
                    elif "gpt2" in request.model.lower():
                        # Some GPT-2 variants need special handling
                        logger.info(f"Trying alternative loading approach for GPT-2 model")
                        import transformers
                        config = transformers.AutoConfig.from_pretrained(request.model)
                        tokenizer = transformers.AutoTokenizer.from_pretrained(request.model)
                        model_class = transformers.AutoModelForCausalLM.from_pretrained(request.model)
                        model = inseq.load_model(model_class, method, tokenizer=tokenizer)
                        logger.info(f"Model loaded successfully using transformers directly")
                    else:
                        # Re-raise if none of our special cases apply
                        raise first_error
                except Exception as second_error:
                    logger.error(f"Failed all model loading attempts. Final error: {second_error}")
                    # If all attempts fail, raise the original error for better troubleshooting
                    raise first_error

            # Patch tokenizer for models that need it
            model = self._patch_tokenizer(model)

            # Move model to appropriate device
            if hasattr(model, 'to'):
                try:
                    model.to(model_device)
                    logger.info(f"Model moved to {model_device}")
                except Exception as e:
                    logger.warning(f"Could not move model to {model_device}: {e}")
                    logger.info(f"Continuing with default device")

            # Move to attribution device if needed and different
            if attr_device != model_device and hasattr(model, 'to'):
                try:
                    model.to(attr_device)
                    logger.info(f"Model moved to {attr_device} for attribution")
                except Exception as e:
                    logger.warning(f"Could not move model to {attr_device}: {e}")

            # Use robust settings for the attribution
            logger.info(f"Performing attribution on input: {request.input_text}")

            # Perform attribution with special error handling and enhanced parallelism
            try:
                # Define warm-up function before using it
                def warm_up_cores(device):
                    # Create a small workload to engage all cores
                    logger.info("Warming up CPU cores before attribution...")
                    a = torch.randn(1000, 1000, device=device)
                    b = torch.randn(1000, 1000, device=device)
                    for _ in range(5):
                        _ = torch.mm(a, b)
                        _ = torch.nn.functional.relu(a)
                    logger.info("Core warm-up complete")

                # Run warm-up with the proper device argument
                warm_up_cores(attr_device)

                # Now run the actual attribution with method-specific parameters
                if method in ['integrated_gradients', 'deep_lift', 'gradient_shap',
                              'discretized_integrated_gradients', 'sequential_integrated_gradients',
                              'layer_integrated_gradients', 'layer_deep_lift']:
                    logger.info(f"Using n_steps={request.n_steps} for {method}")

                    # Check if this is CPU-only operation
                    is_cpu_only = attr_device == 'cpu'

                    # For CPU-only operations that need parallelism, use the parallel_apply strategy
                    if is_cpu_only:
                        logger.info("Using parallel strategy for attribution")

                        # Perform the attribution with enhanced CPU utilization
                        attribution = model.attribute(
                            request.input_text,
                            step_scores=request.step_scores,
                            n_steps=request.n_steps
                        )
                    else:
                        # GPU attribution - use standard approach
                        attribution = model.attribute(
                            request.input_text,
                            step_scores=request.step_scores,
                            n_steps=request.n_steps
                        )
                else:
                    # For methods that don't need n_steps
                    attribution = model.attribute(
                        request.input_text,
                        step_scores=request.step_scores
                    )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA out of memory error: {e}")
                    raise Exception(f"GPU out of memory. Try using force_cpu=True or a smaller model.")
                else:
                    logger.error(f"RuntimeError during attribution: {e}")
                    raise

            # Force garbage collection after attribution
            gc.collect()

            # Save attribution results
            job_output_dir = self.output_dir / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)
            json_path = job_output_dir / 'attribution.json'

            # Save with use_primitives=True to make the JSON more easily consumable by the API
            attribution.save(str(json_path), overwrite=True, use_primitives=True)

            logger.info(f"Attribution results saved to {json_path}")
            return str(json_path)

        except Exception as e:
            logger.error(f"Error during attribution: {e}")
            raise Exception(f"Attribution failed: {str(e)}")

    def _determine_device(self, force_cpu=False):
        """
        Determine the best available device for computation.

        Args:
            force_cpu: Whether to force CPU usage.

        Returns:
            tuple: A tuple of (model_device, attr_device).
        """
        # Import torch here to ensure it's available
        import torch

        if force_cpu:
            return 'cpu', 'cpu'

        if torch.cuda.is_available():
            return 'cuda', 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'cpu'
        else:
            return 'cpu', 'cpu'

    def _patch_tokenizer(self, model):
        """
        Add unk_token_id to tokenizer if it doesn't exist and fix common tokenizer issues.

        Args:
            model: The model to patch.

        Returns:
            The patched model.
        """
        # Fix for missing unk_token
        if model.tokenizer.unk_token_id is None:
            model.tokenizer.unk_token = model.tokenizer.convert_ids_to_tokens(0)
            model.tokenizer.unk_token_id = 0

        # Fix for missing pad_token - use eos_token if needed
        if model.tokenizer.pad_token is None:
            if model.tokenizer.eos_token is not None:
                logger.info(f"Setting pad_token to eos_token: {model.tokenizer.eos_token}")
                model.tokenizer.pad_token = model.tokenizer.eos_token
            else:
                logger.info("Setting default pad_token to '<pad>'")
                model.tokenizer.pad_token = "<pad>"

        return model

    def _validate_method(self, method, available_methods):
        """
        Validate if the requested attribution method is available.

        Args:
            method: The requested method.
            available_methods: A list of available methods.

        Returns:
            str: The validated method.
        """
        # Local import of logger for use in subprocess
        import logging
        logger = logging.getLogger(__name__)

        if method not in available_methods:
            logger.warning(f"Method {method} not in available methods, using {available_methods[0]}")
            return available_methods[0]
        return method