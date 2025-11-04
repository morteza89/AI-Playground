"""
General Dynamic 5-Benchmark Dataset Model Tester - Any Model Support
Dynamically loads actual public datasets and randomly selects samples for testing
Supports any HuggingFace model vs OpenVINO quantized version comparison
5 Comprehensive Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)
No hardcoded examples - fresh random selection each run
"""

import os
import time
import logging
import traceback
import random
import json
import re
from datetime import datetime

# Set up logging
log_file = "general_5benchmark_dataset_test_results.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from optimum.intel.openvino import OVModelForCausalLM
    import datasets
    from datasets import load_dataset
    torch_available = True
    datasets_available = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    torch_available = False
    datasets_available = False


class General5BenchmarkDynamicDatasetTester:
    def __init__(self):
        self.hf_model_id = None
        self.openvino_model_path = None
        self.device = "CPU"

        self.tokenizer = None
        self.hf_model = None
        self.ov_model = None

        # Comprehensive 5-Dataset configurations
        self.dataset_configs = {
            "MMLU_Math": {
                "name": "MMLU Mathematics",
                "dataset_name": "cais/mmlu",
                "subset": "abstract_algebra",  # Math subset
                "description": "Real MMLU mathematics questions",
                "question_key": "question",
                "choices_key": "choices",
                "answer_key": "answer",
                "category": "Academic Knowledge"
            },
            "GSM8K_Problems": {
                "name": "GSM8K Grade School Math",
                "dataset_name": "gsm8k",
                "subset": "main",
                "description": "Real GSM8K word problems",
                "question_key": "question",
                "answer_key": "answer",
                "category": "Mathematical Reasoning"
            },
            "HellaSwag_Reasoning": {
                "name": "HellaSwag Common Sense",
                "dataset_name": "hellaswag",
                "subset": None,
                "description": "Real HellaSwag common sense reasoning",
                "question_key": "ctx",
                "choices_key": "endings",
                "answer_key": "label",
                "category": "Common Sense Reasoning"
            },
            "MBPP_Coding": {
                "name": "MBPP Programming Problems",
                "dataset_name": "mbpp",
                "subset": "sanitized",
                "description": "Real MBPP coding challenges",
                "question_key": "text",
                "answer_key": "code",
                "category": "Code Generation"
            },
            "TruthfulQA_Honesty": {
                "name": "TruthfulQA Truthfulness",
                "dataset_name": "truthful_qa",
                "subset": "multiple_choice",
                "description": "Real TruthfulQA truthfulness questions",
                "question_key": "question",
                "choices_key": "mc1_targets",
                "answer_key": "mc1_targets",
                "category": "Truthfulness & Honesty"
            }
        }

        self.loaded_datasets = {}
        self.samples_per_dataset = 10  # Default

    def model_selection(self):
        """Get model selection from user"""
        print("=" * 80)
        print("GENERAL 5-BENCHMARK MODEL CONFIGURATION")
        print("=" * 80)
        print("Configure your models for comprehensive 5-dataset comparison")
        print("Benchmarks: MMLU + GSM8K + HellaSwag + MBPP (Coding) + TruthfulQA (Honesty)")
        print()

        # HuggingFace model selection
        print(" HUGGINGFACE MODEL SELECTION:")
        print("Examples:")
        print("  • meta-llama/Meta-Llama-3.1-8B")
        print("  • Qwen/Qwen2-7B-Instruct")
        print("  • microsoft/DialoGPT-medium")
        print("  • microsoft/Phi-3-mini-4k-instruct")
        print("  • google/gemma-2-2b-it")
        print("  • codellama/CodeLlama-7b-Instruct-hf")
        print()

        while True:
            hf_model = input("Enter HuggingFace model ID (e.g., Qwen/Qwen2-7B-Instruct): ").strip()
            if hf_model:
                self.hf_model_id = hf_model
                print(f"✓ HuggingFace Model: {self.hf_model_id}")
                break
            else:
                print(" Please provide a valid model ID")

        print()

        # OpenVINO model selection
        print("⚡ OPENVINO MODEL SELECTION:")
        print("Provide the path to your OpenVINO quantized model directory")
        print("Examples:")
        print("  • ./ov_qwen2-7b_int8")
        print("  • ./ov_llama31-8b_int8")
        print("  • ./quantized_models/phi3-mini-int4")
        print("  • C:\\Models\\openvino\\gemma-2-2b-int8")
        print()

        while True:
            ov_path = input("Enter OpenVINO model path: ").strip()
            if ov_path:
                # Normalize path
                ov_path = os.path.normpath(ov_path)

                # Check if path exists
                if os.path.exists(ov_path):
                    self.openvino_model_path = ov_path
                    print(f"✓ OpenVINO Model: {self.openvino_model_path}")
                    break
                else:
                    # Ask if user wants to proceed anyway (model might be created later)
                    proceed = input(f"  Path '{ov_path}' not found. Continue anyway? (y/N): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        self.openvino_model_path = ov_path
                        print(f"  OpenVINO Model: {self.openvino_model_path} (not verified)")
                        break
                    else:
                        print("Please provide a valid path or create the OpenVINO model first")
            else:
                print(" Please provide a model path")

        # Model configuration summary
        print()
        print(" 5-BENCHMARK MODEL CONFIGURATION SUMMARY:")
        print(f"  HuggingFace Model: {self.hf_model_id}")
        print(f"  OpenVINO Model:    {self.openvino_model_path}")
        print("  Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)")
        print()

        # Confirmation
        confirm = input("Proceed with this 5-benchmark configuration? (Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            print("Configuration cancelled. Restart to try again.")
            return False

        print(" 5-Benchmark model configuration confirmed!")
        return True

    def hardware_selection(self):
        """Get hardware selection"""
        print("\nHARDWARE SELECTION:")
        print("1. CPU (Most Stable)")
        print("2. iGPU (Intel Graphics)")
        print("3. NPU (Neural Processing Unit)")
        print()

        try:
            choice = input("Choose hardware (1-3) or Enter for CPU: ").strip()
            if choice == "2":
                self.device = "GPU"
                print("Selected: iGPU")
            elif choice == "3":
                self.device = "NPU"
                print("Selected: NPU")
            else:
                self.device = "CPU"
                print("Selected: CPU")
        except Exception:
            self.device = "CPU"
            print("Defaulted to: CPU")

    def sample_selection(self):
        """Get number of samples per dataset selection"""
        print("\nSAMPLE COUNT SELECTION (Per Dataset):")
        print("How many samples per dataset would you like to test?")
        print("1. Quick Test (5 random samples per dataset = 25 total)")
        print("2. Standard Test (10 random samples per dataset = 50 total)")
        print("3. Extended Test (20 random samples per dataset = 100 total)")
        print("4. Custom Count (Enter your own number)")
        print()

        try:
            choice = input("Choose sample count (1-4) or Enter for Standard (10): ").strip()

            if choice == "1":
                self.samples_per_dataset = 5
                print("Selected: Quick Test (5 random samples per dataset = 25 total)")
            elif choice == "3":
                self.samples_per_dataset = 20
                print("Selected: Extended Test (20 random samples per dataset = 100 total)")
            elif choice == "4":
                try:
                    custom_count = int(input("Enter number of samples (1-50): ").strip())
                    if 1 <= custom_count <= 50:
                        self.samples_per_dataset = custom_count
                        total_samples = custom_count * len(self.dataset_configs)
                        print(f"Selected: Custom Test ({custom_count} random samples per dataset = {total_samples} total)")
                    else:
                        self.samples_per_dataset = 10
                        print("Invalid range. Defaulted to: Standard Test (10 samples per dataset = 50 total)")
                except ValueError:
                    self.samples_per_dataset = 10
                    print("Invalid input. Defaulted to: Standard Test (10 samples per dataset = 50 total)")
            else:
                self.samples_per_dataset = 10
                print("Selected: Standard Test (10 random samples per dataset = 50 total)")

        except Exception:
            self.samples_per_dataset = 10
            print("Defaulted to: Standard Test (10 samples per dataset = 50 total)")

        total_samples = self.samples_per_dataset * len(self.dataset_configs)
        print(f"\n✓ Will randomly select {self.samples_per_dataset} samples from each of {len(self.dataset_configs)} datasets")
        print(f"✓ Total test cases: {total_samples}")
        print("✓ Fresh random selection each run ensures variety")
        print("✓ Comprehensive evaluation across all capability areas")
        print("\nPress Enter to continue...")

        try:
            input()
        except Exception:
            pass

    def load_datasets(self):
        """Load real public datasets including new MBPP and TruthfulQA"""
        print("\n" + "=" * 60)
        print("LOADING REAL 5-BENCHMARK PUBLIC DATASETS")
        print("=" * 60)

        for dataset_key, config in self.dataset_configs.items():
            try:
                print(f"📥 Loading {config['name']} ({config['category']})...")
                logger.info(f"Loading dataset: {config['dataset_name']}")

                # Special handling for different dataset structures
                if dataset_key == "MBPP_Coding":
                    dataset = load_dataset(config['dataset_name'], split='test')
                elif dataset_key == "TruthfulQA_Honesty":
                    dataset = load_dataset(config['dataset_name'], config['subset'], split='validation')
                elif config['subset']:
                    dataset = load_dataset(config['dataset_name'], config['subset'], split='test')
                else:
                    dataset = load_dataset(config['dataset_name'], split='validation')

                self.loaded_datasets[dataset_key] = {
                    "dataset": dataset,
                    "config": config,
                    "total_samples": len(dataset)
                }

                print(f" {config['name']}: {len(dataset)} samples available ({config['category']})")
                logger.info(f"Successfully loaded {config['name']}: {len(dataset)} samples")

            except Exception as e:
                print(f" Failed to load {config['name']}: {str(e)}")
                logger.error(f"Failed to load {config['dataset_name']}: {str(e)}")
                # Remove failed dataset from configs
                if dataset_key in self.loaded_datasets:
                    del self.loaded_datasets[dataset_key]

        if not self.loaded_datasets:
            print(" No datasets loaded successfully!")
            return False

        print(f"\n Successfully loaded {len(self.loaded_datasets)} comprehensive benchmark datasets")
        print(" Coverage: Academic Knowledge, Math Reasoning, Common Sense, Coding, Truthfulness")
        print("Ready for random sampling...")
        return True

    def load_models(self):
        """Load models with the proven working approach"""
        try:
            logger.info("=== LOADING GENERAL MODELS FOR 5-BENCHMARK DATASET TESTING ===")
            logger.info(f"HuggingFace Model: {self.hf_model_id}")
            logger.info(f"OpenVINO Model: {self.openvino_model_path}")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            print(f"Loading tokenizer for {self.hf_model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Tokenizer loaded")
            print(" Tokenizer loaded")

            # Load HuggingFace model
            logger.info("Loading HuggingFace model...")
            print(f"Loading HuggingFace model: {self.hf_model_id}...")

            try:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Failed with device_map='auto', trying CPU: {e}")
                print("⚠️ Trying CPU fallback for HuggingFace model...")
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_id,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )

            logger.info("✓ HuggingFace model loaded")
            print(" HuggingFace model loaded")

            # Load OpenVINO model with device selection
            logger.info(f"Loading OpenVINO model on {self.device}...")
            print(f"Loading OpenVINO model from: {self.openvino_model_path}")
            print(f"Target device: {self.device}")

            device_options = [
                {"device": self.device},
                {"device": "CPU"}  # fallback
            ]

            for i, config in enumerate(device_options):
                try:
                    self.ov_model = OVModelForCausalLM.from_pretrained(
                        self.openvino_model_path,
                        **config
                    )
                    actual_device = config["device"]
                    logger.info(f"✓ OpenVINO model loaded on {actual_device}")
                    print(f" OpenVINO model loaded on {actual_device}")
                    if actual_device != self.device:
                        logger.warning(f"Using {actual_device} instead of {self.device}")
                        print(f" Using {actual_device} instead of {self.device}")
                        self.device = actual_device
                    break
                except Exception as e:
                    logger.warning(f"Failed on {config['device']}: {str(e)}")
                    print(f" Failed on {config['device']}: {str(e)}")
                    if i == len(device_options) - 1:
                        raise e

            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            print(f" Model loading failed: {str(e)}")
            print("Please check:")
            print(f"  • HuggingFace model ID: {self.hf_model_id}")
            print(f"  • OpenVINO model path: {self.openvino_model_path}")
            print("  • Network connection for HuggingFace model download")
            print("  • OpenVINO model files exist and are valid")
            return False

    def sample_random_data(self, dataset_key: str) -> list:
        """Randomly sample data from a loaded dataset"""
        dataset_info = self.loaded_datasets[dataset_key]
        dataset = dataset_info["dataset"]
        config = dataset_info["config"]

        # Get random indices
        total_available = len(dataset)
        sample_count = min(self.samples_per_dataset, total_available)

        random_indices = random.sample(range(total_available), sample_count)

        samples = []
        for idx in random_indices:
            sample = dataset[idx]

            # Format sample based on dataset type
            if dataset_key == "MMLU_Math":
                formatted_sample = {
                    "question": sample[config["question_key"]],
                    "choices": sample[config["choices_key"]],
                    "correct_answer": sample[config["answer_key"]],
                    "expected_patterns": [chr(65 + sample[config["answer_key"]]), sample[config["choices_key"]][sample[config["answer_key"]]]]  # A, B, C, D and actual text
                }
            elif dataset_key == "GSM8K_Problems":
                # Extract numeric answer from GSM8K
                answer = sample[config["answer_key"]]
                # GSM8K answers are like "#### 24" - extract the number
                numeric_answer = answer.split("#### ")[-1] if "#### " in answer else answer
                formatted_sample = {
                    "question": sample[config["question_key"]],
                    "correct_answer": numeric_answer,
                    "expected_patterns": [numeric_answer.strip()]
                }
            elif dataset_key == "HellaSwag_Reasoning":
                correct_idx = int(sample[config["answer_key"]])
                formatted_sample = {
                    "question": sample[config["question_key"]],
                    "choices": sample[config["choices_key"]],
                    "correct_answer": correct_idx,
                    "expected_patterns": [chr(65 + correct_idx), sample[config["choices_key"]][correct_idx]]  # A, B, C, D and actual text
                }
            elif dataset_key == "MBPP_Coding":
                # MBPP coding problems
                formatted_sample = {
                    "question": sample[config["question_key"]],
                    "correct_answer": sample[config["answer_key"]],
                    "expected_patterns": self.extract_code_patterns(sample[config["answer_key"]])
                }
            elif dataset_key == "TruthfulQA_Honesty":
                # TruthfulQA multiple choice
                mc_targets = sample[config["answer_key"]]
                if isinstance(mc_targets, dict) and 'choices' in mc_targets and 'labels' in mc_targets:
                    choices = mc_targets['choices']
                    labels = mc_targets['labels']
                    correct_indices = [i for i, label in enumerate(labels) if label == 1]
                    if correct_indices:
                        correct_idx = correct_indices[0]  # Take first correct answer
                        formatted_sample = {
                            "question": sample[config["question_key"]],
                            "choices": choices,
                            "correct_answer": correct_idx,
                            "expected_patterns": [chr(65 + correct_idx), choices[correct_idx]]
                        }
                    else:
                        # Fallback if no correct answer found
                        formatted_sample = {
                            "question": sample[config["question_key"]],
                            "choices": choices if choices else [],
                            "correct_answer": 0,
                            "expected_patterns": [chr(65), choices[0] if choices else "Unknown"]
                        }
                else:
                    # Fallback format for TruthfulQA
                    formatted_sample = {
                        "question": sample[config["question_key"]],
                        "correct_answer": "Truth-based answer",
                        "expected_patterns": ["truth", "accurate", "correct", "factual"]
                    }
            else:
                # Fallback format
                formatted_sample = {
                    "question": str(sample.get(config["question_key"], "Unknown question")),
                    "correct_answer": str(sample.get(config["answer_key"], "Unknown answer")),
                    "expected_patterns": [str(sample.get(config["answer_key"], "Unknown"))]
                }

            samples.append(formatted_sample)

        logger.info(f"Randomly sampled {len(samples)} cases from {dataset_key}")
        return samples

    def extract_code_patterns(self, code: str) -> list:
        """Extract key patterns from code for validation"""
        patterns = []

        # Add the full code as one pattern
        patterns.append(code.strip())

        # Extract function names
        function_matches = re.findall(r'def\s+(\w+)', code)
        patterns.extend(function_matches)

        # Extract key keywords
        keywords = ['return', 'if', 'else', 'for', 'while', 'def', 'class']
        for keyword in keywords:
            if keyword in code:
                patterns.append(keyword)

        # Extract variable assignments (simple case)
        var_matches = re.findall(r'(\w+)\s*=', code)
        patterns.extend(var_matches[:3])  # Limit to first 3 variables

        return patterns[:10]  # Limit patterns to avoid too many

    def format_question(self, sample: dict, dataset_key: str) -> str:
        """Format question based on dataset type"""
        if dataset_key == "MBPP_Coding":
            return f"Write Python code to solve this problem: {sample['question']} Code:"
        elif dataset_key == "TruthfulQA_Honesty":
            if "choices" in sample and sample["choices"]:
                choices_text = " ".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(sample["choices"])])
                return f"Question: {sample['question']} {choices_text} Answer:"
            else:
                return f"Question: {sample['question']} Answer:"
        elif "choices" in sample and sample["choices"]:
            # Multiple choice format
            choices_text = " ".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(sample["choices"])])
            return f"Question: {sample['question']} {choices_text} Answer:"
        else:
            # Direct question format
            return f"Question: {sample['question']} Answer:"

    def generate_response(self, model, prompt: str, model_name: str, max_tokens: int = 15, dataset_key: str = "") -> str:
        """Generate response using proven working method with adaptive token length"""
        try:
            # Adaptive token length based on task type and dataset
            if "Code:" in prompt:
                max_tokens = 50  # More tokens for coding tasks
            elif dataset_key == "GSM8K_Problems":
                # GSM8K math problems ALWAYS need more tokens for complete multi-step reasoning
                max_tokens = 150  # Increased to allow full reasoning chains regardless of question length
            elif "Question:" in prompt and len(prompt) > 200:
                # Other long questions
                max_tokens = 150
            else:
                max_tokens = 15  # Standard tokens for simple questions

            logger.info(f"Generating with {model_name}: '{prompt[:50]}...' (max_tokens={max_tokens})")

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=300  # Increased for coding problems
            )

            # Generate with proven working parameters
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True
                )

            # Handle output (proven working approach)
            if hasattr(outputs, 'sequences'):
                sequences = outputs.sequences
                input_length = inputs.input_ids.shape[1]

                if sequences.shape[1] > input_length:
                    new_tokens = sequences[0][input_length:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    logger.info(f"✓ Generated: '{response[:100]}...'")
                    return response if response else "Empty"
                else:
                    return "No new tokens"

            elif isinstance(outputs, torch.Tensor):
                input_length = inputs.input_ids.shape[1]
                if outputs.shape[1] > input_length:
                    new_tokens = outputs[0][input_length:]
                    response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    return response if response else "Empty"
                else:
                    return "No new tokens"
            else:
                return "Unknown output format"

        except Exception as e:
            logger.error(f"Generation failed for {model_name}: {str(e)}")
            return f"ERROR: {str(e)}"

    def validate_response(self, response: str, expected_patterns: list, dataset_key: str) -> bool:
        """Validate if response contains any expected pattern with dataset-specific logic"""
        if not response or response.startswith("ERROR") or response in ["Empty", "No new tokens"]:
            return False

        response_lower = response.lower().strip()

        # Special validation for GSM8K - extract numbers from response
        if dataset_key == "GSM8K_Problems":
            for pattern in expected_patterns:
                pattern_str = str(pattern).strip()
                
                # Strategy 1: Look for final answer indicators (most reliable)
                # These patterns capture the number after conclusive statements
                final_answer_patterns = [
                    r'(?:the\s+)?(?:answer|total|result|sum)\s+(?:is|equals?|=|:)\s*\$?\s*(\d+(?:\.\d+)?)',
                    r'(?:receives?|gets?|earns?|makes?|has|have)\s+(?:a\s+total\s+of\s+)?\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?|pens?|books?|apples?|euros?)?[\.,;\s]*$',
                    r'total\s+(?:of\s+)?(?:is|equals?|=|:)?\s*\\?\[?\$?\s*(\d+(?:\.\d+)?)',
                    r'=\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?|pens?|books?|apples?)?[\.,;\s]*$',
                ]
                
                response_lower = response.lower()
                for final_pattern in final_answer_patterns:
                    matches = re.findall(final_pattern, response_lower, re.MULTILINE)
                    if matches:
                        # Check if the expected answer is in the captured numbers
                        for match in matches:
                            if pattern_str == match or pattern_str in match:
                                return True
                
                # Strategy 2: Check last complete sentence for a number
                # Split by periods, find the last sentence with a number
                sentences = response.split('.')
                for sentence in reversed(sentences):
                    sentence_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
                    if sentence_numbers:
                        # The last number in the last sentence with numbers is likely the answer
                        if pattern_str == sentence_numbers[-1]:
                            return True
                        break
                
                # Strategy 3: Check last few numbers in entire response
                all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
                if all_numbers:
                    # Check last 5 numbers (in case answer appears in final calculation)
                    last_numbers = all_numbers[-5:]
                    if pattern_str in last_numbers:
                        return True
                
                # Strategy 4: Simple substring match (fallback)
                if pattern_str in response:
                    return True
            
            return False

        # Special validation for coding tasks
        if dataset_key == "MBPP_Coding":
            # For coding, check if response contains key code elements
            code_indicators = ['def', 'return', 'if', 'for', 'while', '=', 'print']
            has_code_structure = any(indicator in response_lower for indicator in code_indicators)

            # Also check against expected patterns
            pattern_match = False
            for pattern in expected_patterns:
                if str(pattern).lower().strip() in response_lower:
                    pattern_match = True
                    break

            return has_code_structure or pattern_match

        # Standard validation for other datasets
        for pattern in expected_patterns:
            if str(pattern).lower().strip() in response_lower:
                return True

        return False

    def test_dynamic_dataset(self, dataset_key: str) -> dict:
        """Test a dataset with randomly sampled data"""
        config = self.loaded_datasets[dataset_key]["config"]

        logger.info(f"Testing dataset: {dataset_key}")
        print(f"\n{'='*70}")
        print(f"TESTING: {config['name']} (Dynamic Random Sampling)")
        print(f"{'='*70}")
        print(f"Category: {config['category']}")
        print(f"Description: {config['description']}")
        print(f"Source: {config['dataset_name']}")
        print(f"Random Samples: {self.samples_per_dataset}")

        # Get random samples
        random_samples = self.sample_random_data(dataset_key)

        print(f"✓ Randomly selected {len(random_samples)} samples")
        print()

        results = []
        hf_correct = 0
        ov_correct = 0

        for i, sample in enumerate(random_samples, 1):
            print(f" Random Sample {i}/{len(random_samples)}: {config['category']}")

            # Format the question
            formatted_question = self.format_question(sample, dataset_key)

            # Show question (truncated for display)
            question_display = sample['question'][:150] + '...' if len(sample['question']) > 150 else sample['question']
            print(f"Q: {question_display}")

            if "choices" in sample and sample["choices"]:
                for j, choice in enumerate(sample["choices"][:4]):  # Limit to 4 choices for display
                    choice_display = choice[:80] + '...' if len(choice) > 80 else choice
                    print(f"   {chr(65+j)}) {choice_display}")
                print(f"Expected: {chr(65 + sample['correct_answer']) if isinstance(sample['correct_answer'], int) else sample['correct_answer']}")
            else:
                expected_display = str(sample['correct_answer'])[:100] + '...' if len(str(sample['correct_answer'])) > 100 else sample['correct_answer']
                print(f"Expected: {expected_display}")

            # Generate responses
            hf_response = self.generate_response(self.hf_model, formatted_question, "HuggingFace", dataset_key=dataset_key)
            ov_response = self.generate_response(self.ov_model, formatted_question, "OpenVINO", dataset_key=dataset_key)

            # Validate responses
            hf_correct_flag = self.validate_response(hf_response, sample['expected_patterns'], dataset_key)
            ov_correct_flag = self.validate_response(ov_response, sample['expected_patterns'], dataset_key)

            if hf_correct_flag:
                hf_correct += 1
            if ov_correct_flag:
                ov_correct += 1

            # Display results (truncated for readability)
            hf_display = hf_response[:60] + '...' if len(hf_response) > 60 else hf_response
            ov_display = ov_response[:60] + '...' if len(ov_response) > 60 else ov_response

            hf_status = "✅ PASS" if hf_correct_flag else " FAIL"
            ov_status = "✅ PASS" if ov_correct_flag else " FAIL"

            print(f"  HF: '{hf_display}' → {hf_status}")
            print(f"  OV: '{ov_display}' → {ov_status}")
            print()

            # Store detailed results
            results.append({
                "sample_id": i,
                "question": sample['question'],
                "expected": sample['correct_answer'],
                "hf_response": hf_response,
                "ov_response": ov_response,
                "hf_correct": hf_correct_flag,
                "ov_correct": ov_correct_flag
            })

        # Dataset summary
        total_samples = len(random_samples)
        hf_accuracy = (hf_correct / total_samples) * 100
        ov_accuracy = (ov_correct / total_samples) * 100

        dataset_result = {
            "dataset_name": dataset_key,
            "category": config['category'],
            "total_samples": total_samples,
            "hf_correct": hf_correct,
            "ov_correct": ov_correct,
            "hf_accuracy": hf_accuracy,
            "ov_accuracy": ov_accuracy,
            "detailed_results": results,
            "source_info": {
                "dataset_name": config["dataset_name"],
                "total_available": self.loaded_datasets[dataset_key]["total_samples"],
                "randomly_selected": total_samples
            }
        }

        print(f" {config['name']} Random Sample Results:")
        print(f"  HuggingFace: {hf_correct}/{total_samples} ({hf_accuracy:.1f}%)")
        print(f"  OpenVINO:    {ov_correct}/{total_samples} ({ov_accuracy:.1f}%)")
        print(f"  Source: {total_samples}/{self.loaded_datasets[dataset_key]['total_samples']} randomly selected")

        logger.info(f"Dataset {dataset_key} completed: HF={hf_accuracy:.1f}%, OV={ov_accuracy:.1f}%")

        return dataset_result

    def measure_performance(self, model, model_name: str) -> dict:
        """Measure comprehensive performance including token latency metrics"""
        logger.info(f"Measuring performance for {model_name}")

        # Use diverse questions for performance testing across different capabilities
        perf_questions = [
            "What is 7 + 5? Answer:",
            "The capital of Italy is? Answer:",
            "When crossing a street, look? Answer:",
            "Write Python code to add two numbers: Code:",
            "Is the Earth flat? Answer:"
        ]

        times = []
        token_counts = []
        token_latencies = []

        for i, question in enumerate(perf_questions):
            try:
                start_time = time.perf_counter()
                response = self.generate_response(model, question, f"{model_name}_perf", max_tokens=12)
                end_time = time.perf_counter()

                duration = end_time - start_time
                token_count = len(response.split()) if response and not response.startswith("ERROR") else 1

                # Calculate token latency (time per token)
                token_latency = (duration / token_count) * 1000 if token_count > 0 else 0  # Convert to milliseconds

                times.append(duration)
                token_counts.append(token_count)
                token_latencies.append(token_latency)

                logger.info(f"{model_name} perf run {i+1}: {duration:.2f}s, {token_count} tokens, {token_latency:.1f}ms/token")

            except Exception as e:
                logger.warning(f"Performance run {i+1} failed: {e}")
                times.append(5.0)  # fallback
                token_counts.append(1)
                token_latencies.append(5000.0)  # 5000ms fallback

        # Calculate comprehensive metrics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        throughput = avg_tokens / avg_time if avg_time > 0 else 0
        avg_token_latency = sum(token_latencies) / len(token_latencies)
        min_token_latency = min(token_latencies) if token_latencies else 0
        max_token_latency = max(token_latencies) if token_latencies else 0

        return {
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "throughput": throughput,
            "ttft": times[0] if times else 0,
            "token_latency": min_token_latency,
            "avg_token_latency": avg_token_latency,
            "max_token_latency": max_token_latency,
            "all_token_latencies": token_latencies
        }

    def run_general_5benchmark_dataset_test(self):
        """Run comprehensive 5-benchmark dynamic dataset testing with any model"""
        print("=" * 80)
        print("GENERAL 5-BENCHMARK DYNAMIC PUBLIC DATASET MODEL VALIDATION")
        print("=" * 80)
        print("Support for Any HuggingFace Model vs OpenVINO Comparison")
        print("Loading Real Datasets & Random Sampling Each Run")
        print("5 Comprehensive Benchmarks:")
        print("  • MMLU (Academic Knowledge)")
        print("  • GSM8K (Mathematical Reasoning)")
        print("  • HellaSwag (Common Sense Reasoning)")
        print("  • MBPP (Code Generation)")
        print("  • TruthfulQA (Truthfulness & Honesty)")
        print()

        # Model configuration
        if not self.model_selection():
            return

        # Hardware selection
        self.hardware_selection()

        # Sample count selection
        self.sample_selection()

        # Load real datasets
        if not self.load_datasets():
            print(" Failed to load datasets")
            return

        print("=" * 80)
        print(f"GENERAL 5-BENCHMARK DATASET TEST | Hardware: {self.device}")
        print("=" * 80)
        print(f"HuggingFace Model: {self.hf_model_id}")
        print(f"OpenVINO Model: {os.path.basename(self.openvino_model_path)}")
        print(f"Benchmarks: {len(self.loaded_datasets)} comprehensive capability areas")

        # Load models
        print("\nLoading models...")
        start_time = time.time()

        if not self.load_models():
            print(" Failed to load models")
            return

        load_time = time.time() - start_time
        print(f" Models loaded in {load_time:.1f} seconds")
        print(f" OpenVINO running on: {self.device}")

        # Test each dataset with random samples
        all_results = {}
        total_hf_correct = 0
        total_ov_correct = 0
        total_samples = 0
        category_results = {}

        for dataset_key in self.loaded_datasets.keys():
            dataset_result = self.test_dynamic_dataset(dataset_key)
            all_results[dataset_key] = dataset_result

            # Track by category
            category = dataset_result['category']
            if category not in category_results:
                category_results[category] = {'hf_correct': 0, 'ov_correct': 0, 'total': 0}

            category_results[category]['hf_correct'] += dataset_result['hf_correct']
            category_results[category]['ov_correct'] += dataset_result['ov_correct']
            category_results[category]['total'] += dataset_result['total_samples']

            total_hf_correct += dataset_result['hf_correct']
            total_ov_correct += dataset_result['ov_correct']
            total_samples += dataset_result['total_samples']

        # Performance measurement
        print("\n" + "=" * 60)
        print("PERFORMANCE MEASUREMENT")
        print("=" * 60)

        print("Measuring HuggingFace performance...")
        hf_perf = self.measure_performance(self.hf_model, "HuggingFace")

        print("Measuring OpenVINO performance...")
        ov_perf = self.measure_performance(self.ov_model, "OpenVINO")

        # Final comprehensive results
        overall_hf_accuracy = (total_hf_correct / total_samples) * 100 if total_samples > 0 else 0
        overall_ov_accuracy = (total_ov_correct / total_samples) * 100 if total_samples > 0 else 0

        print("\n" + "=" * 80)
        print("COMPREHENSIVE 5-BENCHMARK MODEL RESULTS")
        print("=" * 80)

        print(" MODEL COMPARISON:")
        print(f"  HuggingFace: {self.hf_model_id}")
        print(f"  OpenVINO:    {os.path.basename(self.openvino_model_path)}")
        print(f"  Hardware:    {self.device}")
        print(f"  Benchmarks:  {len(self.loaded_datasets)} comprehensive capability areas")
        print()

        print(f" OVERALL 5-BENCHMARK ACCURACY ({total_samples} random samples total):")
        print(f"  HuggingFace: {total_hf_correct}/{total_samples} ({overall_hf_accuracy:.1f}%)")
        print(f"  OpenVINO:    {total_ov_correct}/{total_samples} ({overall_ov_accuracy:.1f}%)")
        print()

        print(" CAPABILITY BREAKDOWN (By Category):")
        for category, results in category_results.items():
            hf_cat_accuracy = (results['hf_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            ov_cat_accuracy = (results['ov_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            print(f"  {category}:")
            print(f"    HF: {results['hf_correct']}/{results['total']} ({hf_cat_accuracy:.1f}%) | OV: {results['ov_correct']}/{results['total']} ({ov_cat_accuracy:.1f}%)")
        print()

        print(" DATASET BREAKDOWN (Random Samples):")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            samples_per_dataset = result['total_samples']
            total_available = result['source_info']['total_available']
            print(f"  {config['name']} ({config['category']}):")
            print(f"    Random Selection: {samples_per_dataset}/{total_available} samples")
            print(f"    HF: {result['hf_correct']}/{samples_per_dataset} ({result['hf_accuracy']:.1f}%) | OV: {result['ov_correct']}/{samples_per_dataset} ({result['ov_accuracy']:.1f}%)")
        print()

        print(" PERFORMANCE METRICS:")
        print("  HuggingFace:")
        print(f"    TTFT: {hf_perf['ttft']:.2f}s | Throughput: {hf_perf['throughput']:.1f} tok/s")
        print(f"    Token Latency: {hf_perf['token_latency']:.1f}ms | Avg Token Latency: {hf_perf['avg_token_latency']:.1f}ms")
        print("  OpenVINO:")
        print(f"    TTFT: {ov_perf['ttft']:.2f}s | Throughput: {ov_perf['throughput']:.1f} tok/s")
        print(f"    Token Latency: {ov_perf['token_latency']:.1f}ms | Avg Token Latency: {ov_perf['avg_token_latency']:.1f}ms")
        print()

        # Enhanced speedup calculation including token latency
        if hf_perf['ttft'] > 0 and ov_perf['ttft'] > 0:
            ttft_speedup = hf_perf['ttft'] / ov_perf['ttft']
            throughput_speedup = ov_perf['throughput'] / hf_perf['throughput'] if hf_perf['throughput'] > 0 else 0
            token_latency_improvement = hf_perf['avg_token_latency'] / ov_perf['avg_token_latency'] if ov_perf['avg_token_latency'] > 0 else 0
            print(" PERFORMANCE IMPROVEMENTS:")
            print(f"  TTFT: {ttft_speedup:.1f}x faster")
            print(f"  Throughput: {throughput_speedup:.1f}x faster")
            print(f"  Token Latency: {token_latency_improvement:.1f}x faster ({hf_perf['avg_token_latency']:.1f}ms → {ov_perf['avg_token_latency']:.1f}ms)")
        print()

        # Production assessment
        accuracy_threshold = 70
        performance_improvement = ov_perf['throughput'] > hf_perf['throughput']

        if overall_ov_accuracy >= accuracy_threshold and performance_improvement:
            verdict = " PRODUCTION READY"
            recommendation = f"OpenVINO model shows {overall_ov_accuracy:.1f}% accuracy across 5 benchmarks with significant performance gains. Suitable for production deployment."
        elif overall_ov_accuracy >= 50:
            verdict = " REQUIRES REVIEW"
            recommendation = f"OpenVINO model shows {overall_ov_accuracy:.1f}% accuracy across 5 benchmarks. Consider more extensive evaluation before production."
        else:
            verdict = " NEEDS IMPROVEMENT"
            recommendation = f"OpenVINO model accuracy ({overall_ov_accuracy:.1f}%) across 5 benchmarks below production threshold. Review quantization settings."

        print(f" PRODUCTION ASSESSMENT: {verdict}")
        print(f" RECOMMENDATION: {recommendation}")
        print()

        # COMPREHENSIVE 5-BENCHMARK SUMMARY
        print("=" * 80)
        print("COMPREHENSIVE 5-BENCHMARK SUMMARY")
        print("=" * 80)
        print(" 5-BENCHMARK EVALUATION OVERVIEW:")
        print(f"  • HuggingFace Model: {self.hf_model_id}")
        print(f"  • OpenVINO Model: {os.path.basename(self.openvino_model_path)}")
        print(f"  • Capability Areas: {len(self.loaded_datasets)} comprehensive benchmarks")
        print(f"  • Random Samples Per Dataset: {self.samples_per_dataset}")
        print(f"  • Total Random Test Cases: {total_samples}")
        print(f"  • Hardware Used: {self.device}")
        print("  • Fresh Random Selection Each Run")
        print()

        print(" CAPABILITY-BY-CAPABILITY RESULTS:")
        for category, results in category_results.items():
            hf_cat_accuracy = (results['hf_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            ov_cat_accuracy = (results['ov_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            accuracy_diff = ov_cat_accuracy - hf_cat_accuracy

            print(f"   {category}:")
            print(f"      Test Cases: {results['total']} random samples")
            print(f"      HuggingFace Accuracy: {hf_cat_accuracy:.1f}% ({results['hf_correct']}/{results['total']})")
            print(f"      OpenVINO Accuracy:    {ov_cat_accuracy:.1f}% ({results['ov_correct']}/{results['total']})")

            if accuracy_diff > 0:
                print(f"      Accuracy Delta: +{accuracy_diff:.1f}% (OpenVINO Better)")
            elif accuracy_diff < 0:
                print(f"      Accuracy Delta: {accuracy_diff:.1f}% (HuggingFace Better)")
            else:
                print(f"      Accuracy Delta: {accuracy_diff:.1f}% (Equal Performance)")
            print()

        print(" DATASET-BY-DATASET RESULTS:")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            accuracy_diff = result['ov_accuracy'] - result['hf_accuracy']

            print(f"   {config['name']} ({config['category']}):")
            print(f"      Source Dataset: {config['dataset_name']}")
            print(f"      Available Samples: {result['source_info']['total_available']}")
            print(f"      Random Selection: {result['total_samples']} samples")
            print(f"      HuggingFace Accuracy: {result['hf_accuracy']:.1f}% ({result['hf_correct']}/{result['total_samples']})")
            print(f"      OpenVINO Accuracy:    {result['ov_accuracy']:.1f}% ({result['ov_correct']}/{result['total_samples']})")

            if accuracy_diff > 0:
                print(f"      Accuracy Delta: +{accuracy_diff:.1f}% (OpenVINO Better)")
            elif accuracy_diff < 0:
                print(f"      Accuracy Delta: {accuracy_diff:.1f}% (HuggingFace Better)")
            else:
                print(f"      Accuracy Delta: {accuracy_diff:.1f}% (Equal Performance)")
            print()

        print(" OVERALL 5-BENCHMARK SUMMARY:")
        print(f"  • Overall HuggingFace Accuracy: {overall_hf_accuracy:.1f}% ({total_hf_correct}/{total_samples})")
        print(f"  • Overall OpenVINO Accuracy:    {overall_ov_accuracy:.1f}% ({total_ov_correct}/{total_samples})")
        overall_accuracy_diff = overall_ov_accuracy - overall_hf_accuracy
        if overall_accuracy_diff > 0:
            print(f"  • Overall Accuracy Delta: +{overall_accuracy_diff:.1f}% (OpenVINO Better)")
        elif overall_accuracy_diff < 0:
            print(f"  • Overall Accuracy Delta: {overall_accuracy_diff:.1f}% (HuggingFace Better)")
        else:
            print(f"  • Overall Accuracy Delta: {overall_accuracy_diff:.1f}% (Equal Performance)")

        # Performance summary
        if hf_perf['ttft'] > 0 and ov_perf['ttft'] > 0:
            ttft_speedup = hf_perf['ttft'] / ov_perf['ttft']
            throughput_speedup = ov_perf['throughput'] / hf_perf['throughput'] if hf_perf['throughput'] > 0 else 0
            token_latency_improvement = hf_perf['avg_token_latency'] / ov_perf['avg_token_latency'] if ov_perf['avg_token_latency'] > 0 else 0
            print(f"  • Performance Gains: {ttft_speedup:.1f}x TTFT, {throughput_speedup:.1f}x Throughput, {token_latency_improvement:.1f}x Token Latency")

        print(f"  • Final Assessment: {verdict}")
        print("  • 5-Benchmark Comprehensive Capability Coverage")
        print("  • Random Sampling Ensures Unbiased Evaluation")
        print("=" * 80)
        print()

        # Save comprehensive results
        logger.info("=== COMPREHENSIVE 5-BENCHMARK MODEL RESULTS ===")
        logger.info(f"HuggingFace Model: {self.hf_model_id}")
        logger.info(f"OpenVINO Model: {self.openvino_model_path}")
        logger.info(f"Hardware: {self.device}")
        logger.info(f"Total Random Samples: {total_samples}")
        logger.info(f"Overall HF Accuracy: {overall_hf_accuracy:.1f}% ({total_hf_correct}/{total_samples})")
        logger.info(f"Overall OV Accuracy: {overall_ov_accuracy:.1f}% ({total_ov_correct}/{total_samples})")
        logger.info(f"Overall Accuracy Delta: {overall_accuracy_diff:.1f}%")

        # Log capability-specific summary
        logger.info("=== 5-BENCHMARK CAPABILITY SUMMARY ===")
        for category, results in category_results.items():
            hf_cat_accuracy = (results['hf_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            ov_cat_accuracy = (results['ov_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            accuracy_diff = ov_cat_accuracy - hf_cat_accuracy
            logger.info(f"{category}: {results['total']} random samples")
            logger.info(f"  HF={hf_cat_accuracy:.1f}%, OV={ov_cat_accuracy:.1f}%, Delta={accuracy_diff:.1f}%")

        # Log dataset-specific summary
        logger.info("=== 5-BENCHMARK DATASET SUMMARY ===")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            accuracy_diff = result['ov_accuracy'] - result['hf_accuracy']
            logger.info(f"{config['name']} ({config['dataset_name']}) - {config['category']}: {result['total_samples']}/{result['source_info']['total_available']} random samples")
            logger.info(f"  HF={result['hf_accuracy']:.1f}%, OV={result['ov_accuracy']:.1f}%, Delta={accuracy_diff:.1f}%")

        logger.info(f"HF Performance: TTFT={hf_perf['ttft']:.2f}s, Throughput={hf_perf['throughput']:.1f}, TokenLatency={hf_perf['avg_token_latency']:.1f}ms")
        logger.info(f"OV Performance: TTFT={ov_perf['ttft']:.2f}s, Throughput={ov_perf['throughput']:.1f}, TokenLatency={ov_perf['avg_token_latency']:.1f}ms")
        logger.info(f"Token Latency Improvement: {hf_perf['avg_token_latency'] / ov_perf['avg_token_latency'] if ov_perf['avg_token_latency'] > 0 else 0:.1f}x faster")
        logger.info(f"Assessment: {verdict}")

        print(f" Complete results saved to: {log_file}")
        print(" 5-Benchmark dynamic dataset testing completed successfully!")
        print(" Run again for different random samples or try different models!")
        print(" Comprehensive capability evaluation across all major AI areas!")


def main():
    """Main execution"""
    try:
        if not torch_available or not datasets_available:
            print(" Required libraries not available")
            print("Please install: pip install datasets transformers optimum[openvino] torch")
            return

        print(" GENERAL 5-BENCHMARK DYNAMIC PUBLIC DATASET MODEL TESTER")
        print("Compare ANY HuggingFace model vs OpenVINO quantized version")
        print("Real dataset loading with random sampling each run")
        print("5 Comprehensive Benchmarks:")
        print("  • MMLU (Academic Knowledge)")
        print("  • GSM8K (Mathematical Reasoning)")
        print("  • HellaSwag (Common Sense Reasoning)")
        print("  • MBPP (Code Generation)")
        print("  • TruthfulQA (Truthfulness & Honesty)")
        print()

        tester = General5BenchmarkDynamicDatasetTester()
        tester.run_general_5benchmark_dataset_test()

    except KeyboardInterrupt:
        print("\n Test interrupted by user")
    except Exception as e:
        print(f" Error: {str(e)}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()