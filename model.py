import os
import torch
import gc
import json
import itertools
import numpy as np
from copy import deepcopy
from config import CONFIG
from torch.utils.data import DataLoader, Subset
from dataset import MCQARationaleDataset, CurriculumDataset
from trainers import ChoiceContrastiveTrainer, CurriculumChoiceTrainer, DataCollatorWithRawFields
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from evaluate import load
from trl import SFTTrainer
from datasets import load_dataset
from utils import compute_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_difficulty_score(example):
    """
    Calculate difficulty score for curriculum learning.
    Lower scores mean easier questions.
    """
    score = 0

    # Length of the question
    question_length = len(example['question'].split())
    score += min(question_length / 50.0, 1.0) * 0.2
    
    # Length of the reasoning
    if 'reasoning' in example and example['reasoning']:
        reasoning_length = len(example['reasoning'].split())
        score += min(reasoning_length / 100.0, 1.0) * 0.3
    
    # Dataset drawn from
    datasets_difficulty = {
        'preference_pairs': 1,
        'stemQ generation': 1,
        'generation from arxiv': 0.8,
        'generation': 0.5,
        'MathQA': 0.8,
        'sciq': 0.8,
        'Arc-Easy': 0.2,
        'Arc-Challenge': 0.5,
        'medmcqa': 0.8
    }
    score += datasets_difficulty.get(example['dataset'], 0.5) * 0.4
    
    # Keyword complexity
    complex_keywords = ['because', 'however', 'therefore', 'consequently', 'furthermore', 'nevertheless', 'although', 'whereas', 'meanwhile', 'specifically',
                       'since', 'due to', 'after', 'before', 'while', 'thus','unlike', 'rather than','allows', 'enables']
    text_to_check = example['question']
    if 'reasoning' in example and example['reasoning']:
        text_to_check += " " + example['reasoning']
    
    keyword_count = sum(1 for keyword in complex_keywords if keyword in text_to_check.lower())
    score += min(keyword_count / 5.0, 1.0) * 0.1
    return score

def preprocess_for_trl(examples, tokenizer, params):
    "Converts raw Hugging face dataset to ready-to-train, tokenized, PyTorch compatible dataset"
    
    inputs, labels = [], []
    for q, choices, reasoning, ans in zip(examples['question'], examples['choices'], examples.get('reasoning', []), examples['answer']):
        
        prompt = q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(['A','B','C','D'], choices))
        prompt+= "Answer:" + "\n"
        target = (reasoning or "No explanation provided.") + "\n" + "Answer: " + ans
        
        encoded_in = tokenizer(
            prompt, max_length=params['max_input_len'], truncation=True
        )
        encoded_tgt = tokenizer(
            target, max_length=params['max_target_len'], truncation=True
        )
        
        # mask all but final token to align with the evaluation suite
        lab = encoded_tgt['input_ids']
        pad_id = tokenizer.pad_token_id
        mask = [i for i, t in enumerate(lab) if t!=pad_id]
        new_lab = [-100]*len(lab)
        if mask:
            new_lab[mask[-1]] = lab[mask[-1]]
        inputs.append(encoded_in['input_ids'])
        labels.append(new_lab)
    return {'input_ids': inputs, 'labels': labels}

def choice_likelihood_accuracy(model, tokenizer, examples, device="cuda"):

    "Intern evaluation based on likelihood accuracy, similar to the evaluation suite"
    
    accuracy = load("accuracy")
    preds, refs = [], []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = ord(ex["answer"]) - ord("A")
            losses = []
            for text in choices:
                prompt = q + "\n" + text
                tokens = tokenizer(prompt, return_tensors="pt").to(device)
                out = model(**tokens, labels=tokens["input_ids"])
                losses.append(out.loss.item())
            preds.append(int(min(range(len(losses)), key=losses.__getitem__)))
            refs.append(gold_idx)
    return accuracy.compute(predictions=preds, references=refs)

def train(params, run_index=None):
    # Set up run-specific output directory
    run_dir = params['output_directory']
    if run_index is not None:
        run_dir = os.path.join(params['output_directory'], f"run_{run_index}")
    os.makedirs(run_dir, exist_ok=True)
    
    set_seed(params['random_seed'])
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    # prepare tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds = MCQARationaleDataset(
        params['dataset_path'], 'train', tokenizer,
        max_input_len=params['max_input_len'],
        max_target_len=params['max_target_len']
    )
    val_ds = MCQARationaleDataset(
        params['dataset_path'], 'test', tokenizer,
        max_input_len=params['max_input_len'],
        max_target_len=params['max_target_len']
    )


    # CURRICULUM LEARNING IMPLEMENTATION
    
    curriculum_dataset = None
    if params.get('use_curriculum_learning', True):
        print("Setting up curriculum learning...")
        
        # Calculate difficulty scores for training examples
        raw_train_data = train_ds.data
        difficulty_scores = [calculate_difficulty_score(example) for example in raw_train_data]
        
        curriculum_dataset = CurriculumDataset(
            train_ds, 
            difficulty_scores, 
            schedule=params['curriculum_schedule'],
        )
        
        print(f"Curriculum learning enabled with {len(curriculum_dataset.schedule)} stages")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        params['model_name'], torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    if params.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=params['train_epochs'],
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        lr_scheduler_type = "cosine",
        fp16=params.get('fp16', False),
        weight_decay=params.get('weight_decay', 0.01),
        max_grad_norm = 1.0,
        warmup_steps = 500,

        #Logging
        report_to="none",
        push_to_hub=False,
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=params.get('logging_steps', 100),

        #Evaluation
        eval_strategy="steps",
        eval_steps=params.get('eval_steps', 100),  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=params.get('save_steps', 100),
        save_total_limit=3,
        load_best_model_at_end=True,  # CRITICAL: Load best model
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=2,

    )
    data_collator = DataCollatorWithRawFields(tokenizer)
    
    trainer = create_choice_aware_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=args,
        data_collator = data_collator,
        use_curriculum=params.get('use_curriculum_learning', True),
        curriculum_dataset=curriculum_dataset,
        curriculum_schedule = params.get('curriculum_schedule', None),
        use_choice_loss=params.get('use_choice_loss', True),
        loss_type=params.get('choice_loss_type', 'contrastive'),  # 'contrastive' or 'ranking'
        temperature=params.get('choice_temperature', 1.0),
        margin=params.get('choice_margin', 1.0),
        lambda_contrastive=params.get('lambda_contrastive', 1.0),
        lambda_lm=params.get('lambda_lm', 0)
    )


    gc.collect()
    torch.cuda.empty_cache()
    print(f"Starting training for run {run_index}... use_curriculum_learning={params.get('use_curriculum_learning', True)}")
    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    # Save training logs
    os.makedirs(os.path.join(run_dir, "perf"), exist_ok=True)
    with open(os.path.join(run_dir, "perf", "training_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    # Evaluation
    raw_val = val_ds.data 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = choice_likelihood_accuracy(model, tokenizer, raw_val, device=device)
    print(f"Run {run_index} - Final likelihood accuracy: {result['accuracy']}")
    
    # Save evaluation results
    eval_results = {
        "accuracy": result['accuracy'],
        "config": params
    }
    with open(os.path.join(run_dir, "perf", "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return result['accuracy']


def create_choice_aware_trainer(model, tokenizer, train_dataset, val_dataset, args, data_collator, 
                               use_curriculum=True, curriculum_dataset=None, curriculum_schedule = None, 
                               use_choice_loss=True, loss_type="contrastive",
                               temperature=1.0, margin=1.0, lambda_contrastive=1.0, lambda_lm=0):
    """
    Factory function to create the appropriate trainer based on configuration.
    """
    
    if use_curriculum and curriculum_dataset is not None:
        trainer = CurriculumChoiceTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset =val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            #compute_metrics = compute_metrics,
            curriculum_dataset= curriculum_dataset,
            curriculum_schedule = curriculum_schedule,
            use_choice_loss=use_choice_loss,
            loss_type=loss_type,
            temperature=temperature,
            margin=margin,
            lambda_contrastive=lambda_contrastive,
            lambda_lm=lambda_lm,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
    elif use_choice_loss:
        trainer = ChoiceContrastiveTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            #compute_metrics = compute_metrics,
            use_choice_loss=use_choice_loss,
            loss_type=loss_type,
            temperature=temperature,
            margin=margin,
            lambda_contrastive=lambda_contrastive,
            lambda_lm=lambda_lm,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
    else:
        # Standard trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics = compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
    
    return trainer

if __name__ == '__main__':
    train(CONFIG, 15)

class ChoiceContrastiveTrainer(Trainer):
    """
    Custom trainer that uses choice-specific contrastive loss instead of language modeling loss.
    """
    
    def __init__(self, use_choice_loss=True, loss_type="contrastive", temperature=1.0, 
                 margin=1.0, lambda_contrastive=1.0, lambda_lm=0, **kwargs):
        super().__init__(**kwargs)
        self.use_choice_loss = use_choice_loss
        self.loss_type = loss_type  # "contrastive" or "ranking"
        self.temperature = temperature
        self.margin = margin
        self.lambda_contrastive = lambda_contrastive  # Weight for choice loss
        self.lambda_lm = lambda_lm  # Weight for language modeling loss
        
        # Initialize batch tracking
        self.current_epoch = -1
        self.current_batch_in_epoch = 0
        self.eval_batch_counter = 0

    def get_train_dataloader(self):
        """Override to use custom DataLoader that preserves raw fields"""
        from torch.utils.data import DataLoader
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use custom DataLoader that preserves raw fields"""
        from torch.utils.data import DataLoader
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation combining choice-specific loss with optional LM loss.
        """
        def get_model_inputs(inputs_dict):
            """Extract only the fields that the model expects"""
            model_inputs = {}
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in inputs_dict:
                    model_inputs[key] = inputs_dict[key]
            return model_inputs
        
        if not self.use_choice_loss:
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)
        
        # Debug: Check if raw fields are present
        if "raw_question" not in inputs:
            logging.warning("raw_question not found in inputs. Available keys: " + str(list(inputs.keys())))
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)
            
        batch_size = inputs['input_ids'].shape[0]
        if model.training:
            start_idx = self.current_batch_in_epoch * batch_size
            self.current_batch_in_epoch += 1
        else:
            start_idx = self.eval_batch_counter * batch_size
            self.eval_batch_counter += 1
        
        # Build batch_data from raw fields
        batch_data = []
        try:
            for i in range(len(inputs["raw_question"])):
                q = inputs["raw_question"][i]
                c = inputs["raw_choices"][i]
                a = inputs["raw_answer"][i]
                cot = inputs["raw_reasoning"][i]

                if model.training and cot and cot.strip():
                    q = q + "\n" + cot
                
                # Handle potential tensor conversions
                if isinstance(q, torch.Tensor):
                    q = q.item() if q.ndim == 0 else self.processing_class.decode(q, skip_special_tokens=True)
                if isinstance(a, torch.Tensor):
                    a = a.item() if a.ndim == 0 else self.processing_class.decode(a, skip_special_tokens=True)
                
                # Validate that we have valid strings
                if not isinstance(q, str) or not isinstance(a, str):
                    logging.warning(f"Invalid data types: q={type(q)}, a={type(a)}")
                    continue
                    
                # Check for empty strings
                if len(q.strip()) == 0 or len(a.strip()) == 0:
                    logging.warning(f"Empty strings found: q='{q}', a='{a}'")
                    continue
                    
                batch_data.append((q, c, a))
                
        except Exception as e:
            logging.warning(f"Error building batch_data: {e}")
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)
        
        if not batch_data:
            # Fallback to standard loss if we can't get the raw data
            logging.warning("No valid batch data found for choice loss. Using standard LM loss.")
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)
        
        device = next(model.parameters()).device
        
        # Compute choice-specific loss
        #try:
            
        if self.loss_type == "contrastive":
            if model.training:
                choice_loss = choice_contrastive_loss(model, self.processing_class, batch_data, device, self.temperature,self.lambda_lm, False)
            else:
                choice_loss = choice_contrastive_loss(model, self.processing_class, batch_data, device, self.temperature, self.lambda_lm, True)
        
        # Check for invalid loss values
        if torch.isnan(choice_loss) or torch.isinf(choice_loss):
            logging.warning("Invalid choice loss detected, falling back to LM loss")
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        total_loss = choice_loss
        
        
        if return_outputs:
            # Instead of model(**inputs), run the model on your expanded prompts like in loss computation
            with torch.no_grad():
                # Rebuild inputs for model from your batch_data like in choice_contrastive_loss
                prompts = []
                for question, choices, _ in batch_data:
                    prompt = build_prompt(question, choices)
                    prompts.extend([prompt] * 4)
                inputs_for_eval = self.processing_class(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs_for_eval)
            return (total_loss, outputs)
        else:
            return total_loss

    
    def training_step(self, model, inputs, num_items_in_batch):
        """Override to reset batch index at start of each epoch"""
        # Reset batch counter at the start of each epoch
        if self.state.epoch != self.current_epoch:
            self.current_epoch = self.state.epoch
            self.current_batch_in_epoch = 0
            logging.debug(f"Starting epoch {self.current_epoch}")

        max_batches = len(self.get_train_dataloader())
        if self.current_batch_in_epoch >= max_batches:
            self.current_batch_in_epoch = 0
            logging.debug("Reset batch counter due to bounds check")
        
        return super().training_step(model, inputs, num_items_in_batch)

    def evaluate(self, *args, **kwargs):
        self.eval_batch_counter = 0  # Reset before eval loop
        return super().evaluate(*args, **kwargs)

class DataCollatorWithRawFields:
    def __init__(self, tokenizer, model=None, max_length=512):
        self.processing_class = tokenizer
        self.max_length = max_length
        
    def __call__(self, features):
        if DEBUG:
            print(f"DataCollator received {len(features)} features")
            if features:
                print(f"First feature keys: {list(features[0].keys())}")
                print(f"Raw question sample: '{features[0].get('raw_question', 'NOT_FOUND')}'")
        
        # Extract raw fields first
        raw_questions = [f.get("raw_question", "") for f in features]
        raw_choices = [f.get("raw_choices", []) for f in features]
        raw_answers = [f.get("raw_answer", "") for f in features]
        raw_reasoning = [f.get("raw_reasoning", "") for f in features]
        
        # Create batch with tensor fields only
        batch = {}
        
        # Stack tensors properly
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        batch["input_ids"] = torch.stack(input_ids)
        batch["attention_mask"] = torch.stack(attention_masks)
        batch["labels"] = torch.stack(labels)
        
        # Add raw fields back to the batch
        batch["raw_question"] = raw_questions
        batch["raw_choices"] = raw_choices
        batch["raw_answer"] = raw_answers
        batch["raw_reasoning"] = raw_reasoning
        
        return batch

class CurriculumChoiceTrainer(ChoiceContrastiveTrainer):
    """
    Combines curriculum learning with choice‐specific contrastive loss.
    This version’s compute_loss has been adapted to match the exact logic
    in ChoiceContrastiveTrainer, but sourcing its (question, choices, answer)
    from the curriculum_dataset when in training mode.
    """
    def __init__(self, curriculum_dataset=None, curriculum_schedule=None, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_dataset = curriculum_dataset
        # e.g. [0.3, 0.6, 1.0] to split curriculum into three stages
        self.curriculum_schedule = curriculum_schedule or [0.3, 0.6, 1.0]
        self.last_curriculum_stage = -1
        self.current_eval_batch = 0
        # We reuse current_batch_in_epoch / current_epoch from base class

    def get_train_dataloader(self):
        """If a curriculum dataset is provided, use it for training; otherwise fallback."""
        if self.curriculum_dataset is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.curriculum_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            drop_last=False,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Same as base: no curriculum during evaluation."""
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        1) We remove any manual “batch_indices” logic.
        2) We simply iterate inputs['raw_question'], inputs['raw_choices'], inputs['raw_answer']
           exactly as in the base class, because DataLoader already drew from the current subset.
        """
        def get_model_inputs(inputs_dict):
            model_inputs = {}
            for key in ["input_ids", "attention_mask", "labels"]:
                if key in inputs_dict:
                    model_inputs[key] = inputs_dict[key]
            return model_inputs

        if not self.use_choice_loss:
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        if "raw_question" not in inputs:
            logging.warning(
                "CurriculumChoiceTrainer: raw fields missing. Keys: " + str(list(inputs.keys()))
            )
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        batch_size = inputs["input_ids"].shape[0]

        # ── STEP 1: Build batch_data the same way as in ChoiceContrastiveTrainer ──
        batch_data = []
        try:
            for i in range(len(inputs["raw_question"])):
                q = inputs["raw_question"][i]
                c = inputs["raw_choices"][i]
                a = inputs["raw_answer"][i]
                cot = inputs["raw_reasoning"][i]

                if model.training and cot and cot.strip():
                    q = q + "\n" + cot

                # If these are tensors (e.g. token IDs), decode them:
                if isinstance(q, torch.Tensor):
                    q = q.item() if q.ndim == 0 else self.processing_class.decode(q, skip_special_tokens=True)
                if isinstance(a, torch.Tensor):
                    a = a.item() if a.ndim == 0 else self.processing_class.decode(a, skip_special_tokens=True)

                # Skip invalid cases:
                if not isinstance(q, str) or not isinstance(a, str):
                    logging.warning(f"Invalid types: q={type(q)}, a={type(a)}")
                    continue
                if len(q.strip()) == 0 or len(a.strip()) == 0:
                    logging.warning(f"Empty string: q='{q}', a='{a}'")
                    continue

                batch_data.append((q, c, a))

        except Exception as e:
            logging.warning(f"Error building batch_data: {e}")
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        if not batch_data:
            logging.warning("No valid batch_data; falling back to standard LM loss.")
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        # Truncate extra examples if needed (shouldn’t happen because DataLoader enforces batch_size)
        if len(batch_data) > batch_size:
            batch_data = batch_data[:batch_size]

        device = next(model.parameters()).device

        # ── STEP 2: Compute contrastive (or ranking) loss exactly as in the base class ──
        try:
            if self.loss_type == "contrastive":
                if model.training:
                    choice_loss = choice_contrastive_loss(
                        model,
                        self.processing_class,
                        batch_data,
                        device,
                        self.temperature,
                        self.lambda_lm,
                        False,
                    )
                else:
                    choice_loss = choice_contrastive_loss(
                        model,
                        self.processing_class,
                        batch_data,
                        device,
                        self.temperature,
                        self.lambda_lm,
                        True,
                    )
            else:
                choice_loss = choice_ranking_loss(
                    model,
                    self.processing_class,
                    batch_data,
                    device,
                    self.margin
                )

            if torch.isnan(choice_loss) or torch.isinf(choice_loss):
                logging.warning("Choice loss invalid; falling back to standard LM loss.")
                clean_inputs = get_model_inputs(inputs)
                return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        except Exception as e:
            logging.warning(f"Error computing choice loss: {e}. Using standard LM loss.")
            clean_inputs = get_model_inputs(inputs)
            return super().compute_loss(model, clean_inputs, return_outputs, num_items_in_batch)

        total_loss = choice_loss

        # ── STEP 3: return loss (and optionally outputs) ──
        if return_outputs:
            with torch.no_grad():
                prompts = []
                for question, choices, _ in batch_data:
                    prompt = build_prompt(question, choices)
                    # replicate each prompt 4×, one per choice
                    prompts.extend([prompt] * 4)

                inputs_for_eval = self.processing_class(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                outputs = model(**inputs_for_eval)

            return (total_loss, outputs)
        else:
            if model.training:
                # increment for next call
                self.current_batch_in_epoch += 1
            return total_loss

    def training_step(self, model, inputs, num_items_in_batch):
        """
        Before each training step, possibly advance the curriculum stage
        based on how far along we are (global_step / max_steps). Then delegate
        to super().training_step(…).
        """
        if self.curriculum_dataset is not None and self.state.max_steps > 0:
            progress = self.state.global_step / self.state.max_steps
            # Find which stage we should be in given progress
            target_stage = len(self.curriculum_schedule)  # default to last stage
            for i, threshold in enumerate(self.curriculum_schedule):
                if progress < threshold:
                    target_stage = i
                    break

            # Advance curriculum_dataset.current_stage until it matches target_stage
            while target_stage > self.curriculum_dataset.current_stage:
                self.curriculum_dataset.advance_stage()
                # Reset the per‐epoch batch counter and force dataloader refresh
                self.current_batch_in_epoch = 0
                self._train_dataloader = None
                logging.info(f"Advanced to curriculum stage {self.curriculum_dataset.current_stage + 1}")

        # Make sure we don’t exceed the number of batches in this epoch
        max_batches = len(self.get_train_dataloader()) if hasattr(self, "get_train_dataloader") else float("inf")
        if self.current_batch_in_epoch >= max_batches:
            self.current_batch_in_epoch = 0
            logging.info("Reset batch counter due to bounds check")

        # Optionally, log every N steps:
        if self.state.global_step % 100 == 0 and self.curriculum_dataset is not None:
            stage = self.curriculum_dataset.current_stage + 1
            logging.info(f"Step {self.state.global_step}: curriculum stage {stage}")

        return super().training_step(model, inputs, num_items_in_batch)

    def evaluate(self, *args, **kwargs):
        """
        Reset our eval‐batch counter so that indexing for eval always starts at 0.
        """
        self.current_eval_batch = 0
        return super().evaluate(*args, **kwargs)

import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

class MCQARationaleDataset(Dataset):
    def __init__(self, hf_name, split, tokenizer, max_input_len=256, max_target_len=128):
        
        self.data = load_dataset(hf_name, split=split)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        question = example["question"]
        choices = example["choices"]
        choice_letters = ["A", "B", "C", "D"]
        reasoning = example['reasoning'] if example['reasoning'] is not None else ""
        #prompt = question + "\n" + "\n".join(
        #    [f"{letter}. {text}" for letter, text in zip(choice_letters, choices)]
        #)
        prompt = question + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(choice_letters, choices))
        prompt += "\nAnswer:"

        #target = (example["reasoning"] or "") +"\n" + "Answer:" + example['answer']
        target = example['answer']

        # Tokenize input and target
        input_enc = self.tokenizer(
            prompt,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Masking everything except the answer letter
        labels = target_enc["input_ids"].squeeze(0)
        pad_id = self.tokenizer.pad_token_id
        
        mask = labels != pad_id
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        last = idxs[-1].item() if idxs.numel() > 0 else -1
        new_labels = torch.full_like(labels, -100)
        if last >= 0:
            new_labels[last] = labels[last]

        item = {
            "input_ids":input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": new_labels,
            "raw_question": question,
            "raw_choices": choices,
            "raw_answer": target,
            "raw_reasoning":reasoning,
        }
        return item

import numpy as np
from torch.utils.data import Dataset

class CurriculumDataset(Dataset):
    """
    Wraps a base_dataset + difficulty_scores into a “cumulative‐fraction” curriculum.
    `schedule` is a list of floats in ascending order, each in (0, 1], representing
    the cumulative fraction of examples (sorted by difficulty) to expose at each stage.
    
    Example:
      schedule = [0.2, 0.6, 1.0]
      - Stage 0 → easiest 20% of all examples
      - Stage 1 → easiest 60% of all examples
      - Stage 2 → all examples (100%)
    """
    def __init__(self, base_dataset, difficulty_scores, schedule):
        """
        base_dataset: any indexable dataset (len(...) and __getitem__(...))
        difficulty_scores: list or array of floats, len = len(base_dataset).
        schedule: list of floats in (0,1], strictly increasing, and last element must be 1.0.
                  e.g. [0.2, 0.6, 1.0]
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.difficulty_scores = np.array(difficulty_scores, dtype=float)
        
        # Validate schedule
        assert isinstance(schedule, (list, tuple)), "schedule must be a list of floats"
        assert all(0.0 < f <= 1.0 for f in schedule), "each schedule fraction must be in (0,1]"
        assert all(schedule[i] < schedule[i+1] for i in range(len(schedule)-1)), \
            "schedule must be strictly increasing"
        assert abs(schedule[-1] - 1.0) < 1e-6, "last schedule value must be exactly 1.0"
        
        self.schedule = schedule
        self.num_stages = len(schedule)
        self.current_stage = 0
        
        # Sort all indices by ascending difficulty (lowest = “easiest”).
        # np.argsort returns an array of indices; convert to Python ints
        self.sorted_indices = [int(i) for i in np.argsort(self.difficulty_scores)]
        
        # Precompute integer boundaries = [ int(frac * N) for frac in schedule ]
        self.n_samples = len(self.base_dataset)
        self.stage_boundaries = []
        for frac in schedule:
            # round down; ensure at least 1 in stage 0 if frac* N < 1
            idx = int(frac * self.n_samples)
            idx = min(max(idx, 1), self.n_samples)
            self.stage_boundaries.append(idx)
        
        # Quick info prints for debugging
        for i, boundary in enumerate(self.stage_boundaries):
            top_diffs = np.sort(self.difficulty_scores)[boundary - 1]
            print(f"[CurriculumDataset] Stage {i} boundary at index {boundary} "
                  f"(cumulative {schedule[i]*100:.0f}%) → difficulty ≤ {top_diffs:.4f}")
        
        # Initialize current_indices = indices for stage 0
        self.current_indices = self._get_indices_for_stage(0)

    def _get_indices_for_stage(self, stage_idx):
        """
        Return the sorted_indices[: boundary] for that stage.
        """
        boundary = self.stage_boundaries[stage_idx]
        return self.sorted_indices[:boundary]

    def advance_stage(self):
        """
        Move from current_stage to current_stage + 1, if possible.
        """
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            self.current_indices = self._get_indices_for_stage(self.current_stage)
            print(f"[CurriculumDataset] Advanced to stage {self.current_stage} / {self.num_stages-1}, "
                  f"using {len(self.current_indices)} examples ({self.schedule[self.current_stage]*100:.0f}% of data).")
        else:
            # Already at last stage → nothing to do
            print("[CurriculumDataset] Already at final stage; cannot advance further.")

    def __len__(self):
        """
        Length = number of examples available at the current stage.
        """
        return len(self.current_indices)

    def __getitem__(self, idx):
        """
        idx ∈ [0, len(self)). We map that to the “real” index in base_dataset.
        """
        real_idx = self.current_indices[idx]
        return self.base_dataset[real_idx]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer
import logging
import random
from sklearn.metrics import accuracy_score

DEBUG = False

def compute_metrics(eval_pred):
    torch.cuda.empty_cache()

    logits, labels = eval_pred
    if isinstance(logits, tuple):  # 🛠 ensure correct unpacking
        logits = logits[0]

    preds = logits.argmax(axis=1)
    return {'accuracy': accuracy_score(labels, preds)}

def build_prompt(question, choices):
    """Build prompt exactly like the evaluation function to ensure consistency"""
    prompt = question + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(['A','B','C','D'], choices))
    prompt += "\nAnswer:"
    print(prompt)
    return prompt

def choice_contrastive_loss(model, tokenizer, batch, device="cuda", temperature=1.0,lambda_lm = 0, eval_mode=False):
    """
    Batched contrastive loss using log-likelihoods over multiple answer choices.
    """
    if not batch:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    choice_tokens = ["A", "B", "C", "D"]
    prompts = []
    target_ids = []
    gold_indices = []
    
    # Build prompts and target token IDs
    for question, choices, gold_answer in batch:
        # Validate inputs
        if not question or not gold_answer:
            logging.warning(f"Empty question or answer: q='{question}', a='{gold_answer}'")
            continue
            
        
        for token in choice_tokens:
            token_id = tokenizer.encode(token, add_special_tokens=False)
            assert len(token_id) == 1, f"Token {token} split into multiple tokens!"
            target_ids.append(token_id[0])
        
        # Handle gold answer conversion with better error checking
        gold_answer_clean = gold_answer.strip().upper()
        if len(gold_answer_clean) == 0:
            logging.warning(f"Empty gold answer after cleaning: '{gold_answer}'")
            continue
            
        try:
            gold_idx = ord(gold_answer_clean[0]) - ord("A")  # Take first character
            if not (0 <= gold_idx < 4):
                logging.warning(f"Invalid gold index: {gold_idx} for answer: {gold_answer}")
                continue
            gold_indices.append(gold_idx)
        except Exception as e:
            logging.warning(f"Error processing gold answer '{gold_answer}': {e}")
            continue

        prompt = build_prompt(question, choices)
        prompts.extend([prompt] * 4)  # Repeat the same prompt 4 times (one per choice)
        
        if DEBUG:
            print(f"Gold answer: {gold_answer} → index: {gold_idx}")
        if DEBUG and random.random() < 0.1:  # Only print some
            print(f"\nPrompt:\n{prompt}\nChoices: {choices}, Answer: {gold_answer}")
    
    if not prompts or len(gold_indices) == 0:
        logging.warning("No valid prompts or gold indices found")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    if DEBUG:
        print("Gold answer indices:", gold_indices)
    
    # Tokenize prompts (batched)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Append answer tokens
    target_ids_tensor = torch.tensor(target_ids, device=device).unsqueeze(1)  # [batch*4, 1]
    full_input_ids = torch.cat([input_ids, target_ids_tensor], dim=1)
    target_attention = torch.ones_like(target_ids_tensor)
    full_attention_mask = torch.cat([attention_mask, target_attention], dim=1)
    
    # Create labels to only supervise the appended answer token
    labels = full_input_ids.clone()
    labels[:, :-1] = -100  # Only supervise the last token (answer)
    
    with torch.set_grad_enabled(not eval_mode):
        outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask, labels=labels)
        logits = outputs.logits  # [batch*4, seq_len, vocab_size]
        last_token_logits = logits[:, -2, :]  # Logits before answer token
        log_probs = F.log_softmax(last_token_logits / temperature, dim=-1)
        
        # Get log-prob of the answer token
        answer_token_ids = target_ids_tensor.squeeze(1)  # [batch*4]
        selected_log_probs = log_probs[torch.arange(log_probs.size(0)), answer_token_ids]  # [batch*4]
    
    # Reshape into [batch_size, 4]
    batch_size = len(gold_indices)  # Use actual batch size with valid data
    logits_matrix = selected_log_probs.view(batch_size, 4)  # Each row is [logP_A, logP_B, logP_C, logP_D]
    labels = torch.tensor(gold_indices, device=device, dtype=torch.long)
    
    loss = F.cross_entropy(logits_matrix, labels, label_smoothing=0.1)
    
    if lambda_lm > 0:
        # Compute LM loss (e.g., causal language modeling loss)
        lm_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids )
        lm_loss = lm_outputs.loss
    
        # Combine with contrastive or multiple-choice loss
        loss = loss + lambda_lm * lm_loss
    return loss
