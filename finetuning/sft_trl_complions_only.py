import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./llama31_sft_peft_results"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
MAX_LENGTH = 512
EVAL_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

# Get the main split
main_split = "train" if "train" in dataset else list(dataset.keys())[0]
data = dataset[main_split]

# Split into train and eval
train_data, eval_data = train_test_split(data, test_size=EVAL_RATIO, random_state=42)

# BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Prepare model for PEFT
print("Setting up PEFT...")
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Apply PEFT to model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Custom data collator to handle completion-only loss calculation
class CompletionOnlyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = "<|im_start|>assistant\n"
        self.end_token = "<|im_end|>"
        
    def __call__(self, examples):
        batch = {}
        
        # Process each example
        input_ids_list = [example["input_ids"] for example in examples]
        attention_mask_list = [example["attention_mask"] for example in examples]
        
        # Initialize labels with -100 for all tokens (to be ignored in loss calculation)
        labels_list = [[-100] * len(ids) for ids in input_ids_list]
        
        # For each example, find the assistant's response part and set labels for only those tokens
        for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            # Convert input_ids to string to find the assistant token
            text = self.tokenizer.decode(input_ids)
            
            # Find the position of the assistant token
            assistant_pos = text.find(self.assistant_token)
            
            if assistant_pos != -1:
                # Get the text starting from the assistant token
                assistant_text = text[assistant_pos:]
                
                # Find the end of the assistant's message
                end_pos = assistant_text.find(self.end_token)
                if end_pos != -1:
                    assistant_text = assistant_text[:end_pos + len(self.end_token)]
                
                # Convert the assistant text to tokens
                assistant_tokens = self.tokenizer.encode(assistant_text)
                
                # Find the start position of assistant tokens in the input_ids
                start_pos = len(self.tokenizer.encode(text[:assistant_pos]))
                
                # Set labels for the assistant tokens
                for j in range(start_pos, min(start_pos + len(assistant_tokens), len(labels))):
                    labels[j] = input_ids[j]
                    
        # Create the batch
        batch["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_mask_list, dtype=torch.long)
        batch["labels"] = torch.tensor(labels_list, dtype=torch.long)
        
        return batch

# Modified preprocessing function
def preprocess_function(examples):
    # This function handles batch processing
    formatted_texts = []
    
    for i in range(len(examples['instruction']) if 'instruction' in examples else len(examples['text'])):
        if 'instruction' in examples and 'response' in examples:
            if 'input' in examples and examples['input'][i]:
                text = f"<|im_start|>user\n{examples['instruction'][i]}\n\n{examples['input'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['response'][i]}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['response'][i]}<|im_end|>"
        elif 'text' in examples:
            text = examples['text'][i]
        else:
            # Skip examples without required fields
            continue
        
        formatted_texts.append(text)
    
    # Use the tokenizer to encode the texts
    tokenized_inputs = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None  # Return Python lists instead of tensors
    )
    
    return tokenized_inputs

# Apply preprocessing to datasets
print("Preprocessing datasets...")
train_dataset = dataset.from_dict(train_data)
eval_dataset = dataset.from_dict(eval_data)

train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

eval_tokenized = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print(f"Train dataset size: {len(train_tokenized)}")
print(f"Eval dataset size: {len(eval_tokenized)}")

# Create data collator for completion-only loss
data_collator = CompletionOnlyDataCollator(tokenizer)

# Custom trainer for completion-only loss tracking
class CompletionOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to properly handle the completion-only loss.
        Since we're setting labels to -100 for tokens we want to ignore,
        the default loss calculation will already handle this correctly.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# Enhanced loss tracking callback with plotting capability
class LossTrackingCallback:
    def __init__(self):
        self.steps = 0
        self.log_every_steps = 5
        self.train_losses = []
        self.eval_losses = []
        self.step_numbers = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.steps += 1
                self.train_losses.append(logs["loss"])
                
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.step_numbers.append(self.steps)
                    
                    # Print losses every 5 steps
                    if len(self.step_numbers) % (self.log_every_steps // 5) == 0:
                        print(f"Step {self.steps}:")
                        print(f"  Train Completion Loss: {logs['loss']:.4f}")
                        print(f"  Eval Completion Loss: {logs['eval_loss']:.4f}")
                        print("-" * 30)
                        
                        # Update the plot every time we log
                        self.plot_losses()
    
    def plot_losses(self):
        """Create and save a plot of training and evaluation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.step_numbers, [self.train_losses[i-1] for i in self.step_numbers], 'b-', label='Training Completion Loss')
        plt.plot(self.step_numbers, self.eval_losses, 'r-', label='Evaluation Completion Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Completion Losses')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{OUTPUT_DIR}/completion_loss_plot.png")
        plt.close()
        
    def on_train_end(self, args, state, control, **kwargs):
        """Final plot at the end of training"""
        if self.step_numbers:
            self.plot_losses()
            
            # Also create and save a separate plot showing convergence
            plt.figure(figsize=(10, 6))
            plt.plot(self.step_numbers, [self.train_losses[i-1] for i in self.step_numbers], 'b-', label='Training Completion Loss')
            plt.plot(self.step_numbers, self.eval_losses, 'r-', label='Evaluation Completion Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Training and Evaluation Completion Loss Convergence')
            plt.legend()
            plt.grid(True)
            
            # Add horizontal lines for final loss values
            if self.train_losses:
                plt.axhline(y=self.train_losses[-1], color='b', linestyle='--', alpha=0.5)
            if self.eval_losses:
                plt.axhline(y=self.eval_losses[-1], color='r', linestyle='--', alpha=0.5)
                
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/final_completion_convergence_plot.png")
            plt.close()

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    eval_steps=5,  # Evaluate every 5 steps
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=1,  # Log every step to capture all losses
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    save_total_limit=3,
)

# Initialize the custom trainer
print("Initializing CompletionOnlyTrainer...")
trainer = CompletionOnlyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Add enhanced loss tracking callback
loss_tracker = LossTrackingCallback()
trainer.add_callback(loss_tracker)

# Train the model
print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
trainer.save_model(f"{OUTPUT_DIR}/final_model")
model.save_pretrained(f"{OUTPUT_DIR}/peft_model")

# Final loss plot
if loss_tracker.step_numbers:
    # Generate a clear final summary plot
    plt.figure(figsize=(12, 8))
    plt.plot(loss_tracker.step_numbers, [loss_tracker.train_losses[i-1] for i in loss_tracker.step_numbers], 'b-', linewidth=2, label='Training Completion Loss')
    plt.plot(loss_tracker.step_numbers, loss_tracker.eval_losses, 'r-', linewidth=2, label='Evaluation Completion Loss')
    
    # Add trend lines
    if len(loss_tracker.step_numbers) > 1:
        from scipy.interpolate import make_interp_spline
        
        # Smooth out the curves for trend lines
        if len(loss_tracker.step_numbers) > 3:
            x_smooth = np.linspace(min(loss_tracker.step_numbers), max(loss_tracker.step_numbers), 100)
            try:
                spl_train = make_interp_spline(loss_tracker.step_numbers, [loss_tracker.train_losses[i-1] for i in loss_tracker.step_numbers], k=min(3, len(loss_tracker.step_numbers)-1))
                y_smooth_train = spl_train(x_smooth)
                plt.plot(x_smooth, y_smooth_train, 'b--', alpha=0.5, label='Training Trend')
                
                spl_eval = make_interp_spline(loss_tracker.step_numbers, loss_tracker.eval_losses, k=min(3, len(loss_tracker.step_numbers)-1))
                y_smooth_eval = spl_eval(x_smooth)
                plt.plot(x_smooth, y_smooth_eval, 'r--', alpha=0.5, label='Evaluation Trend')
            except:
                # Fall back to simpler approach if spline fails
                pass
    
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Completion Loss', fontsize=14)
    plt.title('Training and Evaluation Completion Loss Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Annotate final values
    if loss_tracker.train_losses and loss_tracker.eval_losses:
        plt.annotate(f'Final train: {loss_tracker.train_losses[-1]:.4f}', 
                     xy=(loss_tracker.step_numbers[-1], loss_tracker.train_losses[-1]), 
                     xytext=(5, 0), textcoords='offset points', fontsize=12)
        
        plt.annotate(f'Final eval: {loss_tracker.eval_losses[-1]:.4f}', 
                     xy=(loss_tracker.step_numbers[-1], loss_tracker.eval_losses[-1]), 
                     xytext=(5, 0), textcoords='offset points', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_completion_loss_comparison.png", dpi=300)
    plt.close()

print("Training complete! Completion loss plots saved to output directory.")
