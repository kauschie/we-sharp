# HOW TO USE:

# # Use default values
# python fad_score.py

# # Specify custom baseline, eval, and model
# python fad_score.py --baseline /path/to/baseline --eval /path/to/eval --model some-model

# # Use FAD-inf method
# python fad_score.py --use-inf

# # Specify number of workers
# python fad_score.py --workers 16

import argparse
import time
from pathlib import Path
from fadtk.fad import FrechetAudioDistance, log
from fadtk.model_loader import get_all_models
from fadtk.fad_batch import cache_embedding_files

def compute_fad(baseline_dir, eval_dir, model_name, workers=8, use_inf=False):
    # Retrieve available models and select the one you want to use.
    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    model = models[model_name]

    # Precompute (cache) the embedding files for both datasets.
    for d in [baseline_dir, eval_dir]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=workers)

    # Create the FAD instance.
    fad = FrechetAudioDistance(model, audio_load_worker=workers, load_model=False)
    
    # Compute the FAD score.
    if use_inf:
        # FAD-inf requires the evaluation dataset to be a directory.
        eval_files = list(Path(eval_dir).glob('*.*'))
        result = fad.score_inf(baseline_dir, eval_files)
        score = result.score
        inf_r2 = result.r2
        log.info("FAD-inf Information: %s", result)
    else:
        score = fad.score(baseline_dir, eval_dir)
        inf_r2 = None

    # Log and return the results
    log.info(f"FAD score computed: {score}")
    return score

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute Frechet Audio Distance (FAD) score')
    parser.add_argument('--baseline', 
                        default='/home/bstitt/we-sharp/data/baseline/testing/',
                        help='Path to baseline audio directory')
    parser.add_argument('--eval', 
                        default='/home/bstitt/we-sharp/data/eval/p1/', 
                        help='Path to evaluation audio directory')
    parser.add_argument('--model', 
                        default='clap-laion-audio', 
                        help='Name of the model to use for FAD computation')
    parser.add_argument('--workers', 
                        type=int, 
                        default=8, 
                        help='Number of workers for parallel processing')
    parser.add_argument('--use-inf', 
                        action='store_true', 
                        help='Use FAD-inf computation method')

    # Parse arguments
    args = parser.parse_args()

    # Compute FAD score
    try:
        score = compute_fad(
            baseline_dir=args.baseline, 
            eval_dir=args.eval, 
            model_name=args.model, 
            workers=args.workers, 
            use_inf=args.use_inf
        )
        
        # Print the score to stdout for direct return
        print(score)
        return score
    
    except Exception as e:
        # Print error to stderr
        print(f"ERROR:{str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()