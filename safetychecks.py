import os
import sys
import signal
import logging
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, pipeline, AutoConfig
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime
from typing import ClassVar
logger = logging.getLogger(__name__)



class SafetyCheck(ABC):
    @abstractmethod
    def check(self, text: str) -> float:
        """
        Return a score in [0,1], where 0 means benign and 1 means malicious.
        """
        pass

    def worker_check(self, input_queue, result_queue):
        def sigterm_handler(signum, frame):
            logger.warning("Child process %d: Received SIGTERM, cleaning up.", os.getpid())
            sys.exit(0)
        signal.signal(signal.SIGTERM, sigterm_handler)
        while True:
            text = input_queue.get()
            if text is None:
                break
            score = self.check(text)
            result_queue.put((self.__class__.__name__, score))

class OnnxModelSafetyCheck(SafetyCheck):
    """
    Uses an ONNX model for sequence classification as a safety check.
    Returns the score from the model's output.
    """
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.model_inputs = [i.name for i in self.ort_session.get_inputs()]
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def check(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Ensure all required inputs are present, including token_type_ids if expected by the model
        ort_inputs = {}
        for input_name in self.model_inputs:
            if input_name in inputs:
                ort_inputs[input_name] = inputs[input_name]
            elif input_name == 'token_type_ids' and 'token_type_ids' not in inputs:
                 # Create dummy token_type_ids if the model expects them but tokenizer didn't provide
                 ort_inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'])
            else:
                raise ValueError(f"Model expects input '{input_name}' but it's not provided by the tokenizer.")


        ort_outs = self.ort_session.run(None, ort_inputs)

        # Assuming the model output is a single tensor with scores
        # You might need to adjust this based on your specific ONNX model's output structure
        probs = ort_outs[0][0] # Assuming binary classification, taking the score for the positive class (index 1)
        return self.softmax(probs)[0]

class LLMGuardSafetyCheck(SafetyCheck):
    """
    Uses llm_guard to detect potential malicious/prompt injection.
    Returns 1.0 if "unsafe," else 0.0.
    """
    def __init__(self, model_path = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=False,  # Model is already in ONNX format
            provider="CPUExecutionProvider"
        )
        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
        )

    def check(self, text: str) -> float:
        results = self.pipe(text)
        label = results[0].get("label", "INJECTION")
        score = results[0].get("score", 0.5)
        if label == "INJECTION":
            return score 
        else:
            return 1-score

class PerplexitySafetyCheck:
    """
    Computes perplexity on the text using an ONNX model that provides pre-computed loss.
    If perplexity > threshold => malicious => 1.0, else 0.0.
    Optimized for performance.
    """
    DEFAULT_STEEPNESS: ClassVar[float] = 0.05
    DEFAULT_MIDPOINT: ClassVar[float] = 85
    def __init__(self, model_path="models/perplexity_model/", threshold=120, onnx_filename="gpt2_with_loss.onnx"):
        self.threshold = threshold
        
        logger.info(f"Initializing PerplexitySafetyCheck: model_path='{model_path}', threshold={threshold}, onnx_filename='{onnx_filename}'")

        try:
            self.config = AutoConfig.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load config or tokenizer from {model_path}. Error: {e}", exc_info=True)
            raise

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ('{self.tokenizer.eos_token}')")
            else:
                # This situation requires adding a new pad token or raising an error.
                # For this optimization, we assume a pad token strategy is resolved.
                logger.error("Tokenizer has no pad_token and no eos_token. Pad token ID checks might be affected.")
        
        self.pad_token_id_value = self.tokenizer.pad_token_id
        self.has_pad_token_id = self.pad_token_id_value is not None

        onnx_full_path = os.path.join(model_path, onnx_filename)
        if not os.path.exists(onnx_full_path):
            logger.error(f"ONNX model file not found at {onnx_full_path}")
            raise FileNotFoundError(f"ONNX model file not found at {onnx_full_path}")
        # Create SessionOptions object
        sess_options = onnxruntime.SessionOptions()

        # Set graph optimization level (e.g., EXTENDED for good balance, ALL for max optimization)
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            # OPTIMIZATION: Select the best available ONNX Runtime Execution Provider.
            # Order matters: try more performant providers first.
            available_providers = onnxruntime.get_available_providers()
            preferred_providers_ordered = []
            # Example for common high-performance providers:
            if 'TensorrtExecutionProvider' in available_providers:
                preferred_providers_ordered.append('TensorrtExecutionProvider')
            if 'CUDAExecutionProvider' in available_providers:
                preferred_providers_ordered.append('CUDAExecutionProvider')
            # Add other providers like OpenVINOExecutionProvider, DmlExecutionProvider as needed
            # preferred_providers_ordered.append('OpenVINOExecutionProvider') 
            preferred_providers_ordered.append('CPUExecutionProvider') # Fallback

            self.ort_session = onnxruntime.InferenceSession(
                onnx_full_path,
                providers=preferred_providers_ordered,
                sess_options=sess_options,
            )
            logger.info(f"ONNX Runtime session initialized for '{onnx_full_path}' with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model or initialize session from {onnx_full_path}. Error: {e}", exc_info=True)
            raise

        self.input_names = ["input_ids", "attention_mask", "labels"]
        self.output_names = ["loss"] # We only need 'loss' for perplexity
        
        self.max_seq_length = self.config.max_position_embeddings # Or a fixed shorter length if appropriate


    def _normalize_perplexity(self, perplexity: float, score_steepness: float, score_midpoint: float ) -> float:
        """Applies the sigmoid normalization."""
        try:
            score = 1 / (1 + np.exp(score_steepness * (perplexity - score_midpoint)))
        except OverflowError:
            score = 1.0
        return score


    def check(self, text: str) -> float:
        # 1. Tokenize input text
        try:
            encoded_tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True, # Pad to the length of this item (after truncation), not global max_length
                truncation=True,
                max_length=self.max_seq_length
            )
        except Exception as e:
            logger.error(f"PerplexitySafetyCheck: Error during tokenization for text \"{text[:30]}...\". Error: {e}", exc_info=True)
            return 1.0 # Mark as malicious on tokenization error

        input_ids = encoded_tokens["input_ids"].astype(np.int64)
        attention_mask = encoded_tokens["attention_mask"].astype(np.int64)
        
        effective_sequence_length = np.sum(attention_mask[0]) # Assuming batch size of 1 from tokenizer

        loss_value = 100.0  # Default to a high loss value

        if effective_sequence_length < 2:
            # logger.warning can be verbose if many short texts are processed.
            # Consider if this specific log is critical for performance vs. debug.
            # For now, keeping it as it indicates an edge case.
            logger.warning(
                f"PerplexitySafetyCheck: Effective sequence length ({effective_sequence_length}) for input "
                f"\"{text[:30]}...\" is less than 2. Using high artificial loss."
            )
            # loss_value remains 100.0
        else:
            labels = np.copy(input_ids)
            if self.has_pad_token_id:
                labels[labels == self.pad_token_id_value] = -100
            ort_inputs = {
                self.input_names[0]: input_ids,
                self.input_names[1]: attention_mask,
                self.input_names[2]: labels
            }

            try:
                # Request only the "loss" output
                ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
                retrieved_loss = ort_outputs[0]

                # Check for valid loss (non-NaN, non-negative)
                if not (np.isnan(retrieved_loss) or retrieved_loss < 0):
                    loss_value = retrieved_loss
                else:
                    logger.warning(
                        f"PerplexitySafetyCheck: Invalid loss value ({retrieved_loss}) from ONNX model "
                        f"for input \"{text[:30]}...\". Using high artificial loss."
                    )
                    # loss_value remains 100.0 (defaulted high)
                # Reduce logging in hot path:
                # logger.debug(f"Retrieved loss: {loss_value:.4f}")

            except Exception as e:
                logger.error(
                    f"PerplexitySafetyCheck: Error during ONNX inference for input \"{text[:30]}...\". "
                    f"Using high artificial loss. Error: {e}", exc_info=True
                )
                # loss_value remains 100.0

        perplexity = np.inf

        try:
            clamped_loss = min(loss_value, 7.0)
            perplexity = np.exp(clamped_loss)
        except OverflowError:
            logger.warning(f"PerplexitySafetyCheck: Math overflow for exp({loss_value}) on input \"{text[:30]}...\". Perplexity set to infinity.")
        except ValueError: # E.g. math.exp(nan) if loss_value became NaN despite prior checks
            logger.warning(f"PerplexitySafetyCheck: Math ValueError for exp({loss_value}) on input \"{text[:30]}...\". Perplexity set to infinity.")


        return self._normalize_perplexity(perplexity, self.DEFAULT_STEEPNESS, self.DEFAULT_MIDPOINT)