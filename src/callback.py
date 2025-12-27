import torch
import numpy as np
from transformers import TrainerCallback, PreTrainedTokenizer

class TranslationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        eval_dataset,
        num_samples: int = 20, # Reduced default for speed
        max_new_tokens: int = 256,
        system_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        self.best_bleu = 0.0
        self.best_meteor = 0.0
        self.best_chrf = 0.0

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after evaluation - compute metrics"""
        if state.is_local_process_zero:

            print(f"📈 Computing Metrics (Step {state.global_step})")

            predictions, references = self._generate(model)

            if not predictions:
                print("⚠️ No predictions generated.")
                return

            scores = compute_metrics(predictions, references)
            
            # Print results
            print(f"BLEU: {scores['bleu']:.2f} | METEOR: {scores['meteor']:.2f} | ChrF: {scores['chrf']:.2f}")
            
            # Track best
            if scores['bleu'] > self.best_bleu:
                self.best_bleu = scores['bleu']
                print(f"⭐ New best BLEU: {self.best_bleu:.2f}")
            
            if scores['meteor'] > self.best_meteor:
                self.best_meteor = scores['meteor']
                print(f"⭐ New best METEOR: {self.best_meteor:.2f}")
            
            if scores['chrf'] > self.best_chrf:
                self.best_chrf = scores['chrf']
                print(f"⭐ New best ChrF: {self.best_chrf:.2f}")
            
            # Log to WandB/TensorBoard
            self._log(scores, state.global_step)
            
            print("=" * 70 + "\n")

    def _generate(self, model):
        """Generate translations on sample data"""
        model.eval()
        device = next(model.parameters()).device # Get correct device reliably
        
        predictions = []
        references = []

        # Sample random indices
        total_len = len(self.eval_dataset)
        num_samples = min(self.num_samples, total_len)
        indices = np.random.choice(total_len, num_samples, replace=False)

        print(f"  Generating {len(indices)} samples...")

        with torch.no_grad():
            for idx in indices:
                sample = self.eval_dataset[int(idx)]
                
                # Handle "messages" format
                if "messages" in sample and isinstance(sample["messages"], list):
                    msgs = sample["messages"]
                else:
                    continue

                # Extract content
                reference = None
                input_msgs = []
                
                for msg in msgs:
                    if msg["role"] == "assistant":
                        reference = msg["content"]
                    elif msg["role"] in ["user", "system"]:
                        input_msgs.append(msg)

                if not reference:
                    continue

                # Insert system prompt if missing
                has_system = any(m["role"] == "system" for m in input_msgs)
                if not has_system and self.system_prompt:
                    input_msgs.insert(0, {"role": "system", "content": self.system_prompt})

                # Prepare inputs
                # Using apply_chat_template to get tensors directly
                input_ids = self.tokenizer.apply_chat_template(
                    input_msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)

                # Generate
                outputs = model.generate(
                    input_ids=input_ids, # Pass explicitly as keyword arg
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.1, # Low temp for translation
                    top_p=0.9,
                    do_sample=False  # Greedy decoding for stability
                )

                # Decode
                # Slice the output to remove the input tokens
                generated_ids = outputs[0][input_ids.shape[1]:]
                prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                predictions.append(prediction)
                # Note: references must be a list of lists for SacreBLEU
                references.append([reference])

        model.train() # Switch back to training mode
        return predictions, references

    def _log(self, scores, step):
        """Log to WandB/TensorBoard"""
        # Try WandB
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "eval/bleu": scores['bleu'],
                    "eval/meteor": scores['meteor'],
                    "eval/chrf": scores['chrf'],
                }, step=step)
        except ImportError:
            pass

        # Try TensorBoard (via transformers trainer callback mechanism usually, but we can try manual)
        try:
            from torch.utils.tensorboard import SummaryWriter
            # Note: This might create a separate log file if not managed by Trainer. 
            # Ideally, we rely on Trainer's logging, but this is a standalone backup.
        except ImportError:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        """Print final summary"""
        print("\n" + "=" * 70)
        print("🏆 BEST SCORES")
        print("=" * 70)
        print(f"BLEU:   {self.best_bleu:.2f}")
        print(f"METEOR: {self.best_meteor:.4f}")
        print(f"ChrF:   {self.best_chrf:.2f}")
        print("=" * 70 + "\n")