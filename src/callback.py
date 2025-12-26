import torch
import numpy as np
from transformers import TrainerCallback, PreTrainedTokenizer
from metrics import compute_metrics

class TranslationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        eval_dataset,
        num_samples: int = 100,
        max_new_token: int = 256,
        system_prompt: str = "",
    ): 
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.max_new_token = max_new_token
        self.system_prompt = system_prompt
        
        self.best_bleu = 0
        self.best_meteor = 0
        self.best_chrf = 0
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after evaluation - compute metrics"""
        
        print("\n" + "=" * 70)
        print(f"📈 Computing Metrics (Step {state.global_step})")
        print("=" * 70)
        
        # Generate predictions
        predictions, references = self._generate(model)
        
        # Compute scores
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
        device = model.device
        
        predictions = []
        references = []
        
        # Sample random indices
        indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)
        
        with torch.no_grad():
            for idx in indices:
                sample = self.eval_dataset[int(idx)]
                messages = sample["messages"]
                
                # Get reference (assistant message)
                reference = None
                for msg in messages:
                    if msg["role"] == "assistant":
                        reference = msg["content"]
                        break
                
                if not reference:
                    continue
                
                # Create prompt (without assistant)
                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                
                # Add system prompt if provided
                if self.system_prompt:
                    # Check if there's already a system message
                    has_system = any(m["role"] == "system" for m in prompt_messages)
                    if not has_system:
                        # Prepend system prompt
                        prompt_messages = [
                            {"role": "system", "content": self.system_prompt}
                        ] + prompt_messages
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_token,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode
                generated = outputs[0][inputs["input_ids"].shape[1]:]
                prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                
                predictions.append(prediction)
                references.append(reference)
        
        return predictions, references
    
    def _log(self, scores, step):
        """Log to WandB/TensorBoard"""
        # Try WandB
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "eval_bleu": scores['bleu'],
                    "eval_meteor": scores['meteor'],
                    "eval_chrf": scores['chrf'],
                }, step=step)
        except:
            pass
        
        # Try TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            writer.add_scalar("eval/bleu", scores['bleu'], step)
            writer.add_scalar("eval/meteor", scores['meteor'], step)
            writer.add_scalar("eval/chrf", scores['chrf'], step)
            writer.close()
        except:
            pass
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print final summary"""
        print("\n" + "=" * 70)
        print("🏆 BEST SCORES")
        print("=" * 70)
        print(f"BLEU:   {self.best_bleu:.2f}")
        print(f"METEOR: {self.best_meteor:.2f}")
        print(f"ChrF:   {self.best_chrf:.2f}")
        print("=" * 70 + "\n")

        