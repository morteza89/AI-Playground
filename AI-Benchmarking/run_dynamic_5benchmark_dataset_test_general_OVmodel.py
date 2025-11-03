"""
OpenVINO Model 5-Benchmark Dataset Tester
Dynamically loads actual public datasets and randomly selects samples for testing
Benchmarks OpenVINO quantized models only (no HuggingFace comparison)
5 Comprehensive Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)
No hardcoded examples - fresh random selection each run
"""

import os
import time
import logging
import traceback
import random
import re

# Set up logging
log_file = "openvino_5benchmark_test_results.log"
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
    from transformers import AutoTokenizer
    from optimum.intel.openvino import OVModelForCausalLM
    from datasets import load_dataset
    torch_available = True
    datasets_available = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    torch_available = False
    datasets_available = False


class OpenVINO5BenchmarkTester:
    def __init__(self):
        self.openvino_model_path = None
        self.device = "CPU"
        self.tokenizer = None
        self.ov_model = None

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

    def model_selection(self):
        """Get OpenVINO model selection from user"""
        print("=" * 80)
        print("OPENVINO 5-BENCHMARK MODEL EVALUATION")
        print("=" * 80)
        print("Benchmark OpenVINO quantized model on 5 comprehensive datasets")
        print("Benchmarks: MMLU + GSM8K + HellaSwag + MBPP (Coding) + TruthfulQA (Honesty)")
        print()

        # OpenVINO model selection
        print("OPENVINO MODEL SELECTION:")
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
                    print(f"OpenVINO Model: {self.openvino_model_path}")
                    break
                else:
                    # Ask if user wants to proceed anyway
                    proceed = input(f"Path '{ov_path}' not found. Continue anyway? (y/N): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        self.openvino_model_path = ov_path
                        print(f"OpenVINO Model: {self.openvino_model_path} (not verified)")
                        break
                    else:
                        print("Please provide a valid path or create the OpenVINO model first")
            else:
                print("Please provide a model path")

        # Model configuration summary
        print()
        print("5-BENCHMARK MODEL CONFIGURATION SUMMARY:")
        print(f"  OpenVINO Model: {self.openvino_model_path}")
        print("  Benchmarks: MMLU, GSM8K, HellaSwag, MBPP (Coding), TruthfulQA (Honesty)")
        print()

        # Confirmation
        confirm = input("Proceed with this 5-benchmark configuration? (Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            print("Configuration cancelled. Restart to try again.")
            return False

        print("5-Benchmark OpenVINO model configuration confirmed")
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

    def load_model(self):
        """Load OpenVINO model with device selection"""
        try:
            logger.info("=== LOADING OPENVINO MODEL FOR 5-BENCHMARK TESTING ===")
            logger.info(f"OpenVINO Model: {self.openvino_model_path}")

            # Use model directory for tokenizer
            tokenizer_path = self.openvino_model_path

            # Try to load tokenizer from the model directory
            logger.info("Loading tokenizer...")
            print(f"Loading tokenizer from {tokenizer_path}...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from model path: {e}")
                print(f"Failed to load tokenizer from {tokenizer_path}")
                print("Please ensure the model directory contains tokenizer files")
                return False

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded")
            print("Tokenizer loaded")

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
                    logger.info(f"OpenVINO model loaded on {actual_device}")
                    print(f"OpenVINO model loaded on {actual_device}")
                    if actual_device != self.device:
                        logger.warning(f"Using {actual_device} instead of {self.device}")
                        print(f"Using {actual_device} instead of {self.device}")
                        self.device = actual_device
                    break
                except Exception as e:
                    logger.warning(f"Failed on {config['device']}: {str(e)}")
                    print(f"Failed on {config['device']}: {str(e)}")
                    if i == len(device_options) - 1:
                        raise e

            return True

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            print(f"Model loading failed: {str(e)}")
            print("Please check:")
            print(f"  • OpenVINO model path: {self.openvino_model_path}")
            print("  • Model files exist and are valid")
            print("  • Tokenizer files are present in the model directory")
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

    def generate_response(self, prompt: str, max_tokens: int = 15) -> str:
        """Generate response using OpenVINO model"""
        try:
            # Adaptive token length based on task type
            if "Code:" in prompt:
                max_tokens = 50
            elif "Question:" in prompt and len(prompt) > 200:
                max_tokens = 25
            else:
                max_tokens = 15

            logger.info(f"Generating response: '{prompt[:50]}...' (max_tokens={max_tokens})")

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=300
            )

            # Generate
            with torch.no_grad():
                outputs = self.ov_model.generate(
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
                    logger.info(f"Generated: '{response[:100]}...'")
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
            logger.error(f"Generation failed: {str(e)}")
            return f"ERROR: {str(e)}"

    def validate_response(self, response: str, expected_patterns: list, dataset_key: str) -> bool:
        """Validate if response contains any expected pattern"""
        if not response or response.startswith("ERROR") or response in ["Empty", "No new tokens"]:
            return False

        response_lower = response.lower().strip()

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

        print(f"Randomly selected {len(random_samples)} samples")
        print()

        results = []
        correct = 0

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

            # Generate response
            response = self.generate_response(formatted_question)

            # Validate response
            is_correct = self.validate_response(response, sample['expected_patterns'], dataset_key)

            if is_correct:
                correct += 1

            # Display result
            response_display = response[:60] + '...' if len(response) > 60 else response
            status = "PASS" if is_correct else "FAIL"

            print(f"  Response: '{response_display}' -> {status}")
            print()

            # Store detailed results
            results.append({
                "sample_id": i,
                "question": sample['question'],
                "expected": sample['correct_answer'],
                "response": response,
                "correct": is_correct
            })

        # Dataset summary
        total_samples = len(random_samples)
        accuracy = (correct / total_samples) * 100

        dataset_result = {
            "dataset_name": dataset_key,
            "category": config['category'],
            "total_samples": total_samples,
            "correct": correct,
            "accuracy": accuracy,
            "detailed_results": results,
            "source_info": {
                "dataset_name": config["dataset_name"],
                "total_available": self.loaded_datasets[dataset_key]["total_samples"],
                "randomly_selected": total_samples
            }
        }

        print(f"{config['name']} Random Sample Results:")
        print(f"  OpenVINO: {correct}/{total_samples} ({accuracy:.1f}%)")
        print(f"  Source: {total_samples}/{self.loaded_datasets[dataset_key]['total_samples']} randomly selected")

        logger.info(f"Dataset {dataset_key} completed: Accuracy={accuracy:.1f}%")

        return dataset_result

    def measure_performance(self) -> dict:
        """Measure comprehensive performance including token latency metrics"""
        logger.info("Measuring OpenVINO model performance")

        # Use diverse questions for performance testing
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
                response = self.generate_response(question, max_tokens=12)
                end_time = time.perf_counter()

                duration = end_time - start_time
                token_count = len(response.split()) if response and not response.startswith("ERROR") else 1

                # Calculate token latency
                token_latency = (duration / token_count) * 1000 if token_count > 0 else 0

                times.append(duration)
                token_counts.append(token_count)
                token_latencies.append(token_latency)

                logger.info(f"Performance run {i+1}: {duration:.2f}s, {token_count} tokens, {token_latency:.1f}ms/token")

            except Exception as e:
                logger.warning(f"Performance run {i+1} failed: {e}")
                times.append(5.0)
                token_counts.append(1)
                token_latencies.append(5000.0)

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

    def run_openvino_5benchmark_test(self):
        """Run comprehensive 5-benchmark testing for OpenVINO model"""
        print("=" * 80)
        print("OPENVINO 5-BENCHMARK DYNAMIC DATASET EVALUATION")
        print("=" * 80)
        print("OpenVINO model benchmarking on real datasets")
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
            print("Failed to load datasets")
            return

        print("=" * 80)
        print(f"OPENVINO 5-BENCHMARK TEST | Hardware: {self.device}")
        print("=" * 80)
        print(f"OpenVINO Model: {os.path.basename(self.openvino_model_path)}")
        print(f"Benchmarks: {len(self.loaded_datasets)} comprehensive capability areas")

        # Load model
        print("\nLoading OpenVINO model...")
        start_time = time.time()

        if not self.load_model():
            print("Failed to load model")
            return

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.1f} seconds")
        print(f"OpenVINO running on: {self.device}")

        # Test each dataset with random samples
        all_results = {}
        total_correct = 0
        total_samples = 0
        category_results = {}

        for dataset_key in self.loaded_datasets.keys():
            dataset_result = self.test_dynamic_dataset(dataset_key)
            all_results[dataset_key] = dataset_result

            # Track by category
            category = dataset_result['category']
            if category not in category_results:
                category_results[category] = {'correct': 0, 'total': 0}

            category_results[category]['correct'] += dataset_result['correct']
            category_results[category]['total'] += dataset_result['total_samples']

            total_correct += dataset_result['correct']
            total_samples += dataset_result['total_samples']

        # Performance measurement
        print("\n" + "=" * 60)
        print("PERFORMANCE MEASUREMENT")
        print("=" * 60)

        print("Measuring OpenVINO model performance...")
        ov_perf = self.measure_performance()

        # Final comprehensive results
        overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

        print("\n" + "=" * 80)
        print("COMPREHENSIVE 5-BENCHMARK OPENVINO RESULTS")
        print("=" * 80)

        print("MODEL EVALUATION:")
        print(f"  OpenVINO Model: {os.path.basename(self.openvino_model_path)}")
        print(f"  Hardware: {self.device}")
        print(f"  Benchmarks: {len(self.loaded_datasets)} comprehensive capability areas")
        print()

        print(f"OVERALL 5-BENCHMARK ACCURACY ({total_samples} random samples total):")
        print(f"  OpenVINO: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")
        print()

        print("CAPABILITY BREAKDOWN (By Category):")
        for category, results in category_results.items():
            cat_accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
            print(f"  {category}:")
            print(f"    Accuracy: {results['correct']}/{results['total']} ({cat_accuracy:.1f}%)")
        print()

        print("DATASET BREAKDOWN (Random Samples):")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            samples_per_dataset = result['total_samples']
            total_available = result['source_info']['total_available']
            print(f"  {config['name']} ({config['category']}):")
            print(f"    Random Selection: {samples_per_dataset}/{total_available} samples")
            print(f"    Accuracy: {result['correct']}/{samples_per_dataset} ({result['accuracy']:.1f}%)")
        print()

        print("PERFORMANCE METRICS:")
        print("  OpenVINO:")
        print(f"    TTFT: {ov_perf['ttft']:.2f}s | Throughput: {ov_perf['throughput']:.1f} tok/s")
        print(f"    Token Latency: {ov_perf['token_latency']:.1f}ms | Avg Token Latency: {ov_perf['avg_token_latency']:.1f}ms")
        print()

        # Production assessment
        accuracy_threshold = 70

        if overall_accuracy >= accuracy_threshold:
            verdict = "PRODUCTION READY"
            recommendation = f"OpenVINO model shows {overall_accuracy:.1f}% accuracy across 5 benchmarks. Suitable for production deployment."
        elif overall_accuracy >= 50:
            verdict = "REQUIRES REVIEW"
            recommendation = f"OpenVINO model shows {overall_accuracy:.1f}% accuracy across 5 benchmarks. Consider more extensive evaluation before production."
        else:
            verdict = "NEEDS IMPROVEMENT"
            recommendation = f"OpenVINO model accuracy ({overall_accuracy:.1f}%) across 5 benchmarks below production threshold. Review quantization settings."

        print(f"PRODUCTION ASSESSMENT: {verdict}")
        print(f"RECOMMENDATION: {recommendation}")
        print()

        # COMPREHENSIVE SUMMARY
        print("=" * 80)
        print("COMPREHENSIVE 5-BENCHMARK SUMMARY")
        print("=" * 80)
        print("5-BENCHMARK EVALUATION OVERVIEW:")
        print(f"  • OpenVINO Model: {os.path.basename(self.openvino_model_path)}")
        print(f"  • Capability Areas: {len(self.loaded_datasets)} comprehensive benchmarks")
        print(f"  • Random Samples Per Dataset: {self.samples_per_dataset}")
        print(f"  • Total Random Test Cases: {total_samples}")
        print(f"  • Hardware Used: {self.device}")
        print("  • Fresh Random Selection Each Run")
        print()

        print("CAPABILITY-BY-CAPABILITY RESULTS:")
        for category, results in category_results.items():
            cat_accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0

            print(f"   {category}:")
            print(f"      Test Cases: {results['total']} random samples")
            print(f"      Accuracy: {cat_accuracy:.1f}% ({results['correct']}/{results['total']})")
            print()

        print("DATASET-BY-DATASET RESULTS:")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]

            print(f"   {config['name']} ({config['category']}):")
            print(f"      Source Dataset: {config['dataset_name']}")
            print(f"      Available Samples: {result['source_info']['total_available']}")
            print(f"      Random Selection: {result['total_samples']} samples")
            print(f"      Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total_samples']})")
            print()

        print("OVERALL 5-BENCHMARK SUMMARY:")
        print(f"  • Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")
        print(f"  • Performance: TTFT={ov_perf['ttft']:.2f}s, Throughput={ov_perf['throughput']:.1f} tok/s")
        print(f"  • Token Latency: {ov_perf['avg_token_latency']:.1f}ms average")
        print(f"  • Final Assessment: {verdict}")
        print("  • 5-Benchmark Comprehensive Capability Coverage")
        print("  • Random Sampling Ensures Unbiased Evaluation")
        print("=" * 80)
        print()

        # Save comprehensive results
        logger.info("=== COMPREHENSIVE 5-BENCHMARK OPENVINO RESULTS ===")
        logger.info(f"OpenVINO Model: {self.openvino_model_path}")
        logger.info(f"Hardware: {self.device}")
        logger.info(f"Total Random Samples: {total_samples}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")

        # Log capability-specific summary
        logger.info("=== 5-BENCHMARK CAPABILITY SUMMARY ===")
        for category, results in category_results.items():
            cat_accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
            logger.info(f"{category}: {results['total']} random samples")
            logger.info(f"  Accuracy={cat_accuracy:.1f}%")

        # Log dataset-specific summary
        logger.info("=== 5-BENCHMARK DATASET SUMMARY ===")
        for dataset_key, result in all_results.items():
            config = self.loaded_datasets[dataset_key]["config"]
            logger.info(f"{config['name']} ({config['dataset_name']}) - {config['category']}: {result['total_samples']}/{result['source_info']['total_available']} random samples")
            logger.info(f"  Accuracy={result['accuracy']:.1f}%")

        logger.info(f"Performance: TTFT={ov_perf['ttft']:.2f}s, Throughput={ov_perf['throughput']:.1f}, TokenLatency={ov_perf['avg_token_latency']:.1f}ms")
        logger.info(f"Assessment: {verdict}")

        print(f"Complete results saved to: {log_file}")
        print("5-Benchmark OpenVINO testing completed successfully")
        print("Run again for different random samples")


def main():
    """Main execution"""
    try:
        if not torch_available or not datasets_available:
            print("Required libraries not available")
            print("Please install: pip install datasets transformers optimum[openvino] torch")
            return

        print("OPENVINO 5-BENCHMARK DYNAMIC DATASET MODEL TESTER")
        print("Benchmark OpenVINO quantized models on real datasets")
        print("Real dataset loading with random sampling each run")
        print("Datasets: MMLU, GSM8K, HellaSwag, MBPP, TruthfulQA")
        print()

        tester = OpenVINO5BenchmarkTester()
        tester.run_openvino_5benchmark_test()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
