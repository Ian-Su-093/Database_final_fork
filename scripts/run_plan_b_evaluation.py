#!/usr/bin/env python3
"""
Plan B (Best-of-N Clause Repair) Evaluation Script

This script runs the complete Plan B evaluation pipeline on Spider dataset:
1. Loads trained ClausePRM for scoring SQL clauses
2. Generates initial SQL queries with Qwen/CodeLlama
3. Identifies faulty clauses using PRM scoring
4. Generates repair candidates for faulty clauses
5. Selects best repairs using oracle selection
6. Reports execution accuracy and partial matching scores

Usage:
    python scripts/run_plan_b_evaluation.py --config configs/plan_b_config.yaml [--limit N]

Requirements:
    - Trained ClausePRM checkpoint (results/prm_checkpoints/best_checkpoint/)
    - Spider dataset (clause_ppo/data/spider/ or processed dataset)
    - GPU with CUDA support (RTX 4090 recommended)
    - PyTorch 1.13.1+cu116 with transformers 4.37.0
"""

import argparse
import os
import sys
import yaml
from datetime import datetime

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'clause_ppo', 'src'))


def setup_environment():
    """Set up clean environment without conda interference."""
    import torch
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Plan B will run on CPU (much slower).")
        print("GPU: Not detected")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")


def load_config(config_path: str) -> dict:
    """Load Plan B configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    return config


def main():
    parser = argparse.ArgumentParser(description="Run Plan B (Best-of-N Clause Repair) Evaluation")
    parser.add_argument('--config', 
                       default='clause_ppo/configs/eval_qwen_config.yaml',
                       help='Path to evaluation configuration YAML file')
    parser.add_argument('--prm_checkpoint', 
                       default='clause_ppo/results/prm_checkpoints/best_checkpoint',
                       help='Path to trained ClausePRM checkpoint')
    parser.add_argument('--spider_dir',
                       default='clause_ppo/data/spider',
                       help='Path to Spider dataset directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples for testing (default: all)')
    parser.add_argument('--output_dir',
                       default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("🚀 PLAN B EVALUATION - Best-of-N Clause Repair")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup environment
    print("Environment Setup:")
    print("-" * 30)
    setup_environment()
    print()
    
    # Load configuration
    print("Configuration:")
    print("-" * 20)
    config = load_config(args.config)
    print()
    
    # Verify required files
    print("Checking Prerequisites:")
    print("-" * 30)
    
    checkpoints = [
        args.prm_checkpoint,
        os.path.join(args.prm_checkpoint, 'adapter_model.safetensors'),
        os.path.join(args.prm_checkpoint, 'score_head.pt')
    ]
    
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            print(f"✅ {checkpoint}")
        else:
            print(f"❌ {checkpoint} - Missing!")
    
    dataset_files = [
        '/home/henrylin0822/coding/SQL/Database_final/clause_ppo/data/processed/original_dataset.json',
        args.spider_dir
    ]
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            print(f"✅ {dataset_file}")
        else:
            print(f"❌ {dataset_file} - Missing!")
    
    print()
    
    # Run Plan B evaluation
    print("Running Plan B Evaluation:")
    print("-" * 35)
    
    try:
        from eval.best_of_n import eval_best_of_n
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Update config with output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"plan_b_evaluation_{timestamp}.jsonl"
        config['paths']['output_file'] = os.path.join(args.output_dir, output_file)
        
        print(f"Output file: {config['paths']['output_file']}")
        print(f"Sample limit: {args.limit or 'All samples'}")
        print()
        
        # Run evaluation
        results = eval_best_of_n(
            config=config,
            spider_dir=args.spider_dir,
            prm_ckpt=args.prm_checkpoint,
            limit=args.limit
        )
        
        print()
        print("=" * 70)
        print("🎉 PLAN B EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {config['paths']['output_file']}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())