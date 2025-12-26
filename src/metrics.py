def compute_metrics(predictions, references):
    try:
        import evaluate
        
        # Compute BLEU
        bleu_metric = evaluate.load("bleu")
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        bleu_score = bleu_result["bleu"] * 100  # Convert to percentage
        
        # Compute METEOR
        try:
            meteor_metric = evaluate.load("meteor")
            meteor_result = meteor_metric.compute(
                predictions=predictions,
                references=references
            )
            meteor_score = meteor_result["meteor"] * 100  # Convert to percentage
        except Exception as e:
            print(f"⚠️  METEOR error (install: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"): {e}")
            meteor_score = 0.0
        
        # Compute ChrF
        chrf_metric = evaluate.load("chrf")
        chrf_result = chrf_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        chrf_score = chrf_result["score"]
        
        return {
            "bleu": bleu_score,
            "meteor": meteor_score,
            "chrf": chrf_score
        }
        
    except Exception as e:
        print(f"⚠️  Error computing metrics: {e}")
        return {"bleu": 0.0, "meteor": 0.0, "chrf": 0.0}
