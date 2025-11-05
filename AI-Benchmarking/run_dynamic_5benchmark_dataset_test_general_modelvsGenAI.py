"""
General Dynamic 5-Benchmark Dataset Model Tester - HuggingFace vs OpenVINO GenAI
Compares HuggingFace models against OpenVINO GenAI library (optimized pipeline)
Dynamically loads actual public datasets and randomly selects samples for testing
5 Comprehensive Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)
No hardcoded examples - fresh random selection each run
"""

import os
import time
import logging
import traceback
import random
import re

# Note: logging will be configured after model selection
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import openvino_genai as ov_genai
    from datasets import load_dataset
    torch_available = True
    genai_available = True
    datasets_available = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    torch_available = False
    genai_available = False
    datasets_available = False


class General5BenchmarkHFvsGenAITester:
    def __init__(self):
        self.hf_model_id = None
        self.openvino_model_path = None
        self.device = "CPU"

        self.tokenizer = None
        self.hf_model = None
        self.genai_pipe = None

        # Comprehensive 5-Dataset configurations
        self.dataset_configs = {
            "MMLU_Math": {
                "name": "MMLU Mathematics",
                "dataset_name": "cais/mmlu",
                "subset": "abstract_algebra",
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
        self.log_file = None  # Will be set after model selection

    def setup_logging(self, model_name: str):
        """Setup logging with organized output structure"""
        # Create outputs directory structure
        outputs_dir = os.path.join("outputs", model_name)
        os.makedirs(outputs_dir, exist_ok=True)

        # Set log file path
        self.log_file = os.path.join(outputs_dir, "general_5benchmark_hf_vs_genai_test_results.log")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # Override any existing configuration
        )

        logger.info(f"Logging initialized for model: {model_name}")
        logger.info(f"Log file: {self.log_file}")

    def model_selection(self):
        """Get model selection from user"""
        print("=" * 80)
        print("HUGGINGFACE vs OPENVINO GENAI 5-BENCHMARK COMPARISON")
        print("=" * 80)
        print("Compare HuggingFace model against OpenVINO GenAI optimized pipeline")
        print("Benchmarks: MMLU + GSM8K + HellaSwag + MBPP (Coding) + TruthfulQA (Honesty)")
        print()

        # HuggingFace model selection
        print("HUGGINGFACE MODEL SELECTION:")
        print("Provide the HuggingFace model ID")
        print("Examples:")
        print("  • Qwen/Qwen2-7B-Instruct")
        print("  • meta-llama/Llama-3.1-8B-Instruct")
        print("  • microsoft/Phi-3-mini-4k-instruct")
        print("  • google/gemma-2-2b-it")
        print()

        while True:
            hf_id = input("Enter HuggingFace model ID: ").strip()
            if hf_id:
                self.hf_model_id = hf_id
                print(f"HuggingFace Model: {self.hf_model_id}")
                break
            else:
                print("Please provide a model ID")

        # OpenVINO GenAI model selection
        print("\nOPENVINO GENAI MODEL SELECTION:")
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
                    print(f"OpenVINO GenAI Model: {self.openvino_model_path}")
                    break
                else:
                    # Ask if user wants to proceed anyway
                    proceed = input(f"Path '{ov_path}' not found. Continue anyway? (y/N): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        self.openvino_model_path = ov_path
                        print(f"OpenVINO GenAI Model: {self.openvino_model_path} (not verified)")
                        break
                    else:
                        print("Please provide a valid path or create the OpenVINO model first")
            else:
                print("Please provide a model path")

        # Model configuration summary
        print()
        print("5-BENCHMARK COMPARISON MODEL CONFIGURATION SUMMARY:")
        print(f"  HuggingFace Model: {self.hf_model_id}")
        print(f"  OpenVINO GenAI Model: {self.openvino_model_path}")
        print(f"  OpenVINO Library: openvino_genai (optimized pipeline)")
        print("  Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)")
        print()

        # Confirmation
        confirm = input("Proceed with this 5-benchmark HF vs GenAI comparison? (Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            print("Configuration cancelled. Restart to try again.")
            return False

        print("5-Benchmark HF vs GenAI comparison configuration confirmed")
        return True

    def hardware_selection(self):
        """Get hardware selection"""
        print("\nHARDWARE SELECTION (for OpenVINO GenAI):")
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
        print(f"\nWill randomly select {self.samples_per_dataset} samples from each of {len(self.dataset_configs)} datasets")
        print(f"Total test cases: {total_samples}")
        print("Fresh random selection each run ensures variety")
        print("Comprehensive evaluation across all capability areas")
        print("\nPress Enter to continue...")

        try:
            input()
        except Exception:
            pass

    def load_datasets(self):
        """Load real public datasets including MBPP and TruthfulQA"""
        print("\n" + "=" * 60)
        print("LOADING REAL 5-BENCHMARK PUBLIC DATASETS")
        print("=" * 60)

        for dataset_key, config in self.dataset_configs.items():
            try:
                print(f"Loading {config['name']} ({config['category']})...")
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

                print(f"{config['name']}: {len(dataset)} samples available ({config['category']})")
                logger.info(f"Successfully loaded {config['name']}: {len(dataset)} samples")

            except Exception as e:
                print(f"Failed to load {config['name']}: {str(e)}")
                logger.error(f"Failed to load {config['dataset_name']}: {str(e)}")
                if dataset_key in self.loaded_datasets:
                    del self.loaded_datasets[dataset_key]

        if not self.loaded_datasets:
            print("No datasets loaded successfully")
            return False

        print(f"\nSuccessfully loaded {len(self.loaded_datasets)} comprehensive benchmark datasets")
        print("Coverage: Academic Knowledge, Math Reasoning, Common Sense, Coding, Truthfulness")
        print("Ready for random sampling...")
        return True

    def load_models(self):
        """Load both HuggingFace and OpenVINO GenAI models"""
        try:
            logger.info("=== LOADING GENERAL MODELS FOR 5-BENCHMARK DATASET TESTING ===")
            logger.info(f"HuggingFace Model: {self.hf_model_id}")
            logger.info(f"OpenVINO GenAI Model: {self.openvino_model_path}")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            print(f"Loading tokenizer from: {self.hf_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, trust_remote_code=True)
            logger.info("✓ Tokenizer loaded")
            print("✓ Tokenizer loaded")

            # Load HuggingFace model
            logger.info("Loading HuggingFace model...")
            print(f"Loading HuggingFace model: {self.hf_model_id}")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✓ HuggingFace model loaded")
            print("✓ HuggingFace model loaded")

            # Load OpenVINO GenAI pipeline
            logger.info("Loading OpenVINO GenAI pipeline...")
            print(f"Loading OpenVINO GenAI pipeline from: {self.openvino_model_path}")
            print(f"Target device: {self.device}")

            device_options = [
                self.device,
                "CPU"  # fallback
            ]

            for i, device in enumerate(device_options):
                try:
                    logger.info(f"Attempting to load on {device}...")
                    self.genai_pipe = ov_genai.LLMPipeline(self.openvino_model_path, device)

                    logger.info(f"✓ OpenVINO GenAI pipeline loaded on {device}")
                    print(f"✓ OpenVINO GenAI pipeline loaded on {device}")

                    if device != self.device:
                        logger.warning(f"Using {device} instead of {self.device}")
                        print(f"Using {device} instead of {self.device}")
                        self.device = device
                    break
                except Exception as e:
                    logger.warning(f"Failed on {device}: {str(e)}")
                    print(f"Failed on {device}: {str(e)}")
                    if i == len(device_options) - 1:
                        raise e

            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            print(f"Model loading failed: {str(e)}")
            print("Please check:")
            print(f"  • HuggingFace model ID: {self.hf_model_id}")
            print(f"  • OpenVINO GenAI model path: {self.openvino_model_path}")
            print("  • Model files exist and are valid")
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
                    "expected_patterns": [chr(65 + sample[config["answer_key"]]), sample[config["choices_key"]][sample[config["answer_key"]]]]
                }
            elif dataset_key == "GSM8K_Problems":
                answer = sample[config["answer_key"]]
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
                    "expected_patterns": [chr(65 + correct_idx), sample[config["choices_key"]][correct_idx]]
                }
            elif dataset_key == "MBPP_Coding":
                formatted_sample = {
                    "question": sample[config["question_key"]],
                    "correct_answer": sample[config["answer_key"]],
                    "expected_patterns": self.extract_code_patterns(sample[config["answer_key"]])
                }
            elif dataset_key == "TruthfulQA_Honesty":
                mc_targets = sample[config["answer_key"]]
                if isinstance(mc_targets, dict) and 'choices' in mc_targets and 'labels' in mc_targets:
                    choices = mc_targets['choices']
                    labels = mc_targets['labels']
                    correct_indices = [i for i, label in enumerate(labels) if label == 1]
                    if correct_indices:
                        correct_idx = correct_indices[0]
                        formatted_sample = {
                            "question": sample[config["question_key"]],
                            "choices": choices,
                            "correct_answer": correct_idx,
                            "expected_patterns": [chr(65 + correct_idx), choices[correct_idx]]
                        }
                    else:
                        formatted_sample = {
                            "question": sample[config["question_key"]],
                            "choices": choices if choices else [],
                            "correct_answer": 0,
                            "expected_patterns": [chr(65), choices[0] if choices else "Unknown"]
                        }
                else:
                    formatted_sample = {
                        "question": sample[config["question_key"]],
                        "correct_answer": "Truth-based answer",
                        "expected_patterns": ["truth", "accurate", "correct", "factual"]
                    }
            else:
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
        patterns.append(code.strip())

        # Extract function names
        function_matches = re.findall(r'def\s+(\w+)', code)
        patterns.extend(function_matches)

        # Extract key keywords
        keywords = ['return', 'if', 'else', 'for', 'while', 'def', 'class']
        for keyword in keywords:
            if keyword in code:
                patterns.append(keyword)

        # Extract variable assignments
        var_matches = re.findall(r'(\w+)\s*=', code)
        patterns.extend(var_matches[:3])

        return patterns[:10]

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
            choices_text = " ".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(sample["choices"])])
            return f"Question: {sample['question']} {choices_text} Answer:"
        else:
            return f"Question: {sample['question']} Answer:"

    def generate_hf_response(self, prompt: str, max_tokens: int = 15, dataset_key: str = "") -> str:
        """Generate response using HuggingFace model"""
        try:
            # Adaptive token length based on task type and dataset
            if "Code:" in prompt:
                max_tokens = 50
            elif dataset_key == "GSM8K_Problems":
                # GSM8K math problems ALWAYS need more tokens for complete multi-step reasoning
                max_tokens = 150
            elif "Question:" in prompt and len(prompt) > 200:
                # Other long questions
                max_tokens = 150
            else:
                max_tokens = 15

            logger.info(f"Generating with HuggingFace: '{prompt[:50]}...' (max_tokens={max_tokens})")

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=300
            )

            # Move to same device as model
            inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True
                )

            # Handle output
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
            logger.error(f"HF Generation failed: {str(e)}")
            return f"ERROR: {str(e)}"

    def generate_genai_response(self, prompt: str, max_tokens: int = 15, dataset_key: str = "") -> str:
        """Generate response using OpenVINO GenAI pipeline"""
        try:
            # Adaptive token length based on task type and dataset
            if "Code:" in prompt:
                max_tokens = 50
            elif dataset_key == "GSM8K_Problems":
                # GSM8K math problems ALWAYS need more tokens for complete multi-step reasoning
                max_tokens = 150
            elif "Question:" in prompt and len(prompt) > 200:
                # Other long questions
                max_tokens = 150
            else:
                max_tokens = 15

            logger.info(f"Generating with OpenVINO GenAI: '{prompt[:50]}...' (max_tokens={max_tokens})")

            # Configure generation parameters
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.do_sample = False
            config.num_beams = 1

            # Generate using GenAI pipeline
            response = self.genai_pipe.generate(prompt, config)

            # Extract only the new tokens (remove prompt)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            logger.info(f"✓ Generated: '{response[:100]}...'")
            return response if response else "Empty"

        except Exception as e:
            logger.error(f"GenAI Generation failed: {str(e)}")
            return f"ERROR: {str(e)}"

    def validate_response(self, response: str, expected_patterns: list, dataset_key: str) -> bool:
        """Validate if response contains any expected pattern"""
        if not response or response.startswith("ERROR") or response in ["Empty", "No new tokens"]:
            return False

        response_lower = response.lower().strip()

        # Special validation for GSM8K - extract numbers from response
        if dataset_key == "GSM8K_Problems":
            for pattern in expected_patterns:
                pattern_str = str(pattern).strip()

                # Strategy 1: Look for final answer indicators (most reliable)
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
                        for match in matches:
                            if pattern_str == match or pattern_str in match:
                                return True

                # Strategy 2: Check last complete sentence for a number
                sentences = response.split('.')
                for sentence in reversed(sentences):
                    sentence_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
                    if sentence_numbers:
                        if pattern_str == sentence_numbers[-1]:
                            return True
                        break

                # Strategy 3: Check last few numbers in entire response
                all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
                if all_numbers:
                    last_numbers = all_numbers[-5:]
                    if pattern_str in last_numbers:
                        return True

                # Strategy 4: Simple substring match (fallback)
                if pattern_str in response:
                    return True

            return False

        # Special validation for coding tasks
        if dataset_key == "MBPP_Coding":
            code_indicators = ['def', 'return', 'if', 'for', 'while', '=', 'print']
            has_code_structure = any(indicator in response_lower for indicator in code_indicators)

            pattern_match = False
            for pattern in expected_patterns:
                if str(pattern).lower().strip() in response_lower:
                    pattern_match = True
                    break

            return has_code_structure or pattern_match

        # Standard validation
        for pattern in expected_patterns:
            if str(pattern).lower().strip() in response_lower:
                return True

        return False

    def test_dynamic_dataset(self, dataset_key: str) -> dict:
        """Test a dataset with randomly sampled data - comparing HF vs GenAI"""
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

        print(f"Randomly selected {len(random_samples)} samples")
        print()

        results = []
        hf_correct = 0
        genai_correct = 0

        for i, sample in enumerate(random_samples, 1):
            print(f"Random Sample {i}/{len(random_samples)}: {config['category']}")

            # Format the question
            formatted_question = self.format_question(sample, dataset_key)

            # Show question
            question_display = sample['question'][:150] + '...' if len(sample['question']) > 150 else sample['question']
            print(f"Q: {question_display}")

            if "choices" in sample and sample["choices"]:
                for j, choice in enumerate(sample["choices"][:4]):
                    choice_display = choice[:80] + '...' if len(choice) > 80 else choice
                    print(f"   {chr(65+j)}) {choice_display}")
                print(f"Expected: {chr(65 + sample['correct_answer']) if isinstance(sample['correct_answer'], int) else sample['correct_answer']}")
            else:
                expected_display = str(sample['correct_answer'])[:100] + '...' if len(str(sample['correct_answer'])) > 100 else sample['correct_answer']
                print(f"Expected: {expected_display}")

            # Generate responses from both models
            hf_response = self.generate_hf_response(formatted_question, dataset_key=dataset_key)
            genai_response = self.generate_genai_response(formatted_question, dataset_key=dataset_key)

            # Validate responses
            hf_correct_flag = self.validate_response(hf_response, sample['expected_patterns'], dataset_key)
            genai_correct_flag = self.validate_response(genai_response, sample['expected_patterns'], dataset_key)

            if hf_correct_flag:
                hf_correct += 1
            if genai_correct_flag:
                genai_correct += 1

            # Display results
            hf_response_display = hf_response[:60] + '...' if len(hf_response) > 60 else hf_response
            genai_response_display = genai_response[:60] + '...' if len(genai_response) > 60 else genai_response

            hf_status = "PASS" if hf_correct_flag else "FAIL"
            genai_status = "PASS" if genai_correct_flag else "FAIL"

            print(f"  HF: '{hf_response_display}' -> {hf_status}")
            print(f"  GenAI: '{genai_response_display}' -> {genai_status}")
            print()

            # Store detailed results
            results.append({
                "sample_id": i,
                "question": sample['question'],
                "expected": sample['correct_answer'],
                "hf_response": hf_response,
                "genai_response": genai_response,
                "hf_correct": hf_correct_flag,
                "genai_correct": genai_correct_flag
            })

        # Dataset summary
        total_samples = len(random_samples)
        hf_accuracy = (hf_correct / total_samples) * 100
        genai_accuracy = (genai_correct / total_samples) * 100
        delta = genai_accuracy - hf_accuracy

        dataset_result = {
            "dataset_name": dataset_key,
            "category": config['category'],
            "total_samples": total_samples,
            "hf_correct": hf_correct,
            "genai_correct": genai_correct,
            "hf_accuracy": hf_accuracy,
            "genai_accuracy": genai_accuracy,
            "delta": delta,
            "detailed_results": results,
            "source_info": {
                "dataset_name": config["dataset_name"],
                "total_available": self.loaded_datasets[dataset_key]["total_samples"],
                "randomly_selected": total_samples
            }
        }

        print(f"{config['name']} Random Sample Results:")
        print(f"  HuggingFace: {hf_correct}/{total_samples} ({hf_accuracy:.1f}%)")
        print(f"  OpenVINO GenAI: {genai_correct}/{total_samples} ({genai_accuracy:.1f}%)")
        print(f"  Delta: {delta:+.1f}%")
        print(f"  Source: {total_samples}/{self.loaded_datasets[dataset_key]['total_samples']} randomly selected")

        logger.info(f"Dataset {dataset_key} completed: HF={hf_accuracy:.1f}%, GenAI={genai_accuracy:.1f}%, Delta={delta:+.1f}%")

        return dataset_result

    def measure_performance(self) -> dict:
        """Measure comprehensive performance for both models"""
        logger.info("Measuring model performance")

        # Use diverse questions for performance testing
        perf_questions = [
            "What is 7 + 5? Answer:",
            "The capital of Italy is? Answer:",
            "When crossing a street, look? Answer:",
            "Write Python code to add two numbers: Code:",
            "Is the Earth flat? Answer:"
        ]

        hf_times = []
        hf_token_counts = []
        hf_token_latencies = []

        genai_times = []
        genai_token_counts = []
        genai_token_latencies = []

        for i, question in enumerate(perf_questions):
            try:
                # HuggingFace performance
                start_time = time.perf_counter()
                hf_response = self.generate_hf_response(question, max_tokens=12)
                end_time = time.perf_counter()

                duration = end_time - start_time
                token_count = len(hf_response.split()) if hf_response and not hf_response.startswith("ERROR") else 1
                token_latency = (duration / token_count) * 1000 if token_count > 0 else 0

                hf_times.append(duration)
                hf_token_counts.append(token_count)
                hf_token_latencies.append(token_latency)

                logger.info(f"HF Performance run {i+1}: {duration:.2f}s, {token_count} tokens, {token_latency:.1f}ms/token")

            except Exception as e:
                logger.warning(f"HF Performance run {i+1} failed: {e}")
                hf_times.append(10.0)
                hf_token_counts.append(1)
                hf_token_latencies.append(10000.0)

            try:
                # GenAI performance
                start_time = time.perf_counter()
                genai_response = self.generate_genai_response(question, max_tokens=12)
                end_time = time.perf_counter()

                duration = end_time - start_time
                token_count = len(genai_response.split()) if genai_response and not genai_response.startswith("ERROR") else 1
                token_latency = (duration / token_count) * 1000 if token_count > 0 else 0

                genai_times.append(duration)
                genai_token_counts.append(token_count)
                genai_token_latencies.append(token_latency)

                logger.info(f"GenAI Performance run {i+1}: {duration:.2f}s, {token_count} tokens, {token_latency:.1f}ms/token")

            except Exception as e:
                logger.warning(f"GenAI Performance run {i+1} failed: {e}")
                genai_times.append(5.0)
                genai_token_counts.append(1)
                genai_token_latencies.append(5000.0)

        # Calculate comprehensive metrics
        hf_avg_time = sum(hf_times) / len(hf_times)
        hf_avg_tokens = sum(hf_token_counts) / len(hf_token_counts)
        hf_throughput = hf_avg_tokens / hf_avg_time if hf_avg_time > 0 else 0
        hf_avg_token_latency = sum(hf_token_latencies) / len(hf_token_latencies)

        genai_avg_time = sum(genai_times) / len(genai_times)
        genai_avg_tokens = sum(genai_token_counts) / len(genai_token_counts)
        genai_throughput = genai_avg_tokens / genai_avg_time if genai_avg_time > 0 else 0
        genai_avg_token_latency = sum(genai_token_latencies) / len(genai_token_latencies)

        speedup = hf_avg_token_latency / genai_avg_token_latency if genai_avg_token_latency > 0 else 0

        return {
            "hf": {
                "avg_time": hf_avg_time,
                "avg_tokens": hf_avg_tokens,
                "throughput": hf_throughput,
                "ttft": hf_times[0] if hf_times else 0,
                "token_latency": hf_avg_token_latency
            },
            "genai": {
                "avg_time": genai_avg_time,
                "avg_tokens": genai_avg_tokens,
                "throughput": genai_throughput,
                "ttft": genai_times[0] if genai_times else 0,
                "token_latency": genai_avg_token_latency
            },
            "speedup": speedup
        }

    def run_hf_vs_genai_5benchmark_test(self):
        """Run comprehensive 5-benchmark testing comparing HF vs OpenVINO GenAI"""
        print("=" * 80)
        print("HUGGINGFACE vs OPENVINO GENAI 5-BENCHMARK COMPARISON")
        print("=" * 80)
        print("Compare HuggingFace model against OpenVINO GenAI optimized pipeline")
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

        # Setup logging with organized output structure
        model_name = os.path.basename(self.openvino_model_path)
        self.setup_logging(model_name)

        # Hardware selection
        self.hardware_selection()

        # Sample count selection
        self.sample_selection()

        # Load real datasets
        if not self.load_datasets():
            print("Failed to load datasets")
            return

        print("=" * 80)
        print(f"HF vs GENAI 5-BENCHMARK TEST | Hardware: {self.device}")
        print("=" * 80)
        print(f"HuggingFace Model: {self.hf_model_id}")
        print(f"OpenVINO GenAI Model: {os.path.basename(self.openvino_model_path)}")
        print(f"Benchmarks: {len(self.loaded_datasets)} comprehensive capability areas")

        # Load models
        print("\nLoading models...")
        start_time = time.time()

        if not self.load_models():
            print("Failed to load models")
            return

        load_time = time.time() - start_time
        print(f"Models loaded in {load_time:.1f} seconds")
        print(f"OpenVINO GenAI running on: {self.device}")

        # Test each dataset with random samples
        all_results = {}
        hf_total_correct = 0
        genai_total_correct = 0
        total_samples = 0
        category_results = {}

        for dataset_key in self.loaded_datasets.keys():
            dataset_result = self.test_dynamic_dataset(dataset_key)
            all_results[dataset_key] = dataset_result

            # Track by category
            category = dataset_result['category']
            if category not in category_results:
                category_results[category] = {'hf_correct': 0, 'genai_correct': 0, 'total': 0}

            category_results[category]['hf_correct'] += dataset_result['hf_correct']
            category_results[category]['genai_correct'] += dataset_result['genai_correct']
            category_results[category]['total'] += dataset_result['total_samples']

            hf_total_correct += dataset_result['hf_correct']
            genai_total_correct += dataset_result['genai_correct']
            total_samples += dataset_result['total_samples']

        # Performance measurement
        print("\n" + "=" * 60)
        print("PERFORMANCE MEASUREMENT")
        print("=" * 60)

        print("Measuring performance for both models...")
        perf_metrics = self.measure_performance()

        # Final comprehensive results
        hf_overall_accuracy = (hf_total_correct / total_samples) * 100 if total_samples > 0 else 0
        genai_overall_accuracy = (genai_total_correct / total_samples) * 100 if total_samples > 0 else 0
        overall_delta = genai_overall_accuracy - hf_overall_accuracy

        print("\n" + "=" * 80)
        print("COMPREHENSIVE 5-BENCHMARK HF vs GENAI RESULTS")
        print("=" * 80)

        print("MODEL COMPARISON:")
        print(f"  HuggingFace Model: {self.hf_model_id}")
        print(f"  OpenVINO GenAI Model: {os.path.basename(self.openvino_model_path)}")
        print(f"  Hardware: {self.device}")
        print(f"  Benchmarks: {len(self.loaded_datasets)} comprehensive capability areas")
        print()

        print(f"OVERALL 5-BENCHMARK ACCURACY ({total_samples} random samples total):")
        print(f"  HuggingFace: {hf_total_correct}/{total_samples} ({hf_overall_accuracy:.1f}%)")
        print(f"  OpenVINO GenAI: {genai_total_correct}/{total_samples} ({genai_overall_accuracy:.1f}%)")
        print(f"  Delta: {overall_delta:+.1f}%")
        print()

        print("CAPABILITY BREAKDOWN (By Category):")
        for category, results in category_results.items():
            hf_cat_accuracy = (results['hf_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            genai_cat_accuracy = (results['genai_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            cat_delta = genai_cat_accuracy - hf_cat_accuracy
            print(f"  {category}:")
            print(f"    HF: {results['hf_correct']}/{results['total']} ({hf_cat_accuracy:.1f}%)")
            print(f"    GenAI: {results['genai_correct']}/{results['total']} ({genai_cat_accuracy:.1f}%)")
            print(f"    Delta: {cat_delta:+.1f}%")
        print()

        print("DATASET BREAKDOWN (Random Samples):")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            samples_per_dataset = result['total_samples']
            total_available = result['source_info']['total_available']
            print(f"  {config['name']} ({config['category']}):")
            print(f"    Random Selection: {samples_per_dataset}/{total_available} samples")
            print(f"    HF: {result['hf_correct']}/{samples_per_dataset} ({result['hf_accuracy']:.1f}%)")
            print(f"    GenAI: {result['genai_correct']}/{samples_per_dataset} ({result['genai_accuracy']:.1f}%)")
            print(f"    Delta: {result['delta']:+.1f}%")
        print()

        print("PERFORMANCE METRICS:")
        print(f"  HuggingFace: TTFT={perf_metrics['hf']['ttft']:.2f}s, Throughput={perf_metrics['hf']['throughput']:.1f} tok/s, TokenLatency={perf_metrics['hf']['token_latency']:.1f}ms")
        print(f"  OpenVINO GenAI: TTFT={perf_metrics['genai']['ttft']:.2f}s, Throughput={perf_metrics['genai']['throughput']:.1f} tok/s, TokenLatency={perf_metrics['genai']['token_latency']:.1f}ms")
        print(f"  GenAI Speedup: {perf_metrics['speedup']:.1f}x faster")
        print()

        # Production assessment
        accuracy_threshold = 70

        if genai_overall_accuracy >= accuracy_threshold:
            verdict = " PRODUCTION READY"
        elif genai_overall_accuracy >= 50:
            verdict = " REQUIRES REVIEW"
        else:
            verdict = " NEEDS IMPROVEMENT"

        print(f"ASSESSMENT:{verdict}")
        print()

        # Log comprehensive summary
        logger.info("=== COMPREHENSIVE 5-BENCHMARK HF vs GENAI RESULTS ===")
        logger.info(f"HuggingFace Model: {self.hf_model_id}")
        logger.info(f"OpenVINO GenAI Model: {self.openvino_model_path}")
        logger.info(f"Hardware: {self.device}")
        logger.info(f"Total Random Samples: {total_samples}")
        logger.info(f"Overall: HF={hf_overall_accuracy:.1f}%, GenAI={genai_overall_accuracy:.1f}%, Delta={overall_delta:+.1f}%")

        logger.info("=== 5-BENCHMARK CAPABILITY SUMMARY ===")
        for category, results in category_results.items():
            hf_cat_accuracy = (results['hf_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            genai_cat_accuracy = (results['genai_correct'] / results['total']) * 100 if results['total'] > 0 else 0
            cat_delta = genai_cat_accuracy - hf_cat_accuracy
            logger.info(f"{category}: {results['total']} random samples")
            logger.info(f"  HF={hf_cat_accuracy:.1f}%, GenAI={genai_cat_accuracy:.1f}%, Delta={cat_delta:+.1f}%")

        logger.info("=== 5-BENCHMARK DATASET SUMMARY ===")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            logger.info(f"{config['name']} ({config['dataset_name']}) - {config['category']}: {result['total_samples']}/{result['source_info']['total_available']} random samples")
            logger.info(f"  HF={result['hf_accuracy']:.1f}%, GenAI={result['genai_accuracy']:.1f}%, Delta={result['delta']:+.1f}%")

        logger.info(f"HF Performance: TTFT={perf_metrics['hf']['ttft']:.2f}s, Throughput={perf_metrics['hf']['throughput']:.1f}, TokenLatency={perf_metrics['hf']['token_latency']:.1f}ms")
        logger.info(f"GenAI Performance: TTFT={perf_metrics['genai']['ttft']:.2f}s, Throughput={perf_metrics['genai']['throughput']:.1f}, TokenLatency={perf_metrics['genai']['token_latency']:.1f}ms")
        logger.info(f"Token Latency Improvement: {perf_metrics['speedup']:.1f}x faster")
        logger.info(f"Assessment:{verdict}")

        print(f"Complete results saved to: {self.log_file}")
        print("5-Benchmark HF vs GenAI testing completed successfully")
        print("Run again for different random samples")


def main():
    """Main execution"""
    try:
        if not torch_available or not genai_available or not datasets_available:
            print("Required libraries not available")
            print("Please install:")
            print("  pip install torch transformers openvino-genai datasets")
            return

        print("HUGGINGFACE vs OPENVINO GENAI 5-BENCHMARK COMPARISON TESTER")
        print("Compare HuggingFace models against OpenVINO GenAI pipeline")
        print("Real dataset loading with random sampling each run")
        print("Datasets: MMLU, GSM8K, HellaSwag, MBPP, TruthfulQA")
        print()

        tester = General5BenchmarkHFvsGenAITester()
        tester.run_hf_vs_genai_5benchmark_test()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
