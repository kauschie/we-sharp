"""
Frechet Audio Distance Calculator

Usage:
  Default values:  python fad.py
  Custom settings: python fad.py --baseline /path/to/baseline --eval /path/to/eval --model model-name
  FAD-inf method:  python fad.py --use-inf
  Set workers:     python fad.py --workers 16
"""

import argparse
from pathlib import Path
from fadtk.fad import FrechetAudioDistance, log
from fadtk.model_loader import get_all_models
from fadtk.fad_batch import cache_embedding_files


def compute_fad(baseline_dir, eval_dir, model_name, workers=8, use_inf=False):
    # Get available models and validate selection
    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    model = models[model_name]

    # Cache embeddings for performance
    for d in [baseline_dir, eval_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=workers)

    # Initialize FAD calculator
    fad = FrechetAudioDistance(model, audio_load_worker=workers, load_model=False)
    
    # Calculate score using requested method
    if use_inf:
        eval_files = list(Path(eval_dir).glob('*.*'))
        result = fad.score_inf(baseline_dir, eval_files)
        score = result.score
        log.info("FAD-inf Information: %s", result)
    else:
        score = fad.score(baseline_dir, eval_dir)

    log.info(f"FAD score computed: {score}")
    return score


def main():
    parser = argparse.ArgumentParser(description='Compute Frechet Audio Distance (FAD) score')
    parser.add_argument('--baseline', default='./data/baseline/testing/', 
                        help='Path to baseline audio directory')
    parser.add_argument('--eval', default='./data/eval/p1', 
                        help='Path to evaluation audio directory')
    parser.add_argument('--model', default='clap-laion-audio', 
                        help='Name of the model to use for FAD computation')
    parser.add_argument('--workers', type=int, default=8, 
                        help='Number of workers for parallel processing')
    parser.add_argument('--use-inf', action='store_true', 
                        help='Use FAD-inf computation method')

    args = parser.parse_args()

    try:
        score = compute_fad(
            baseline_dir=args.baseline, 
            eval_dir=args.eval, 
            model_name=args.model, 
            workers=args.workers, 
            use_inf=args.use_inf
        )
        
        print(score)
        return score
    
    except Exception as e:
        print(f"ERROR:{str(e)}")
        raise


if __name__ == "__main__":
    main()