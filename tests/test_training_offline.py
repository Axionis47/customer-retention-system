"""Offline integration tests for all training scripts.

Tests that all training scripts can:
1. Parse command-line arguments correctly
2. Load experiment config properly
3. Load data from local paths
4. Initialize models
5. Run at least one training step

This ensures everything works before pushing to GCP.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def test_sft_training():
    """Test SFT training script."""
    print("\n" + "="*60)
    print("TEST 1: SFT Training")
    print("="*60)
    
    # Create a minimal test config
    test_config = {
        "global": {"seed": 42},
        "sft": {
            "base_model": "facebook/opt-350m",
            "output_dir": "test_output/sft",
            "max_steps": 2,  # Just 2 steps for testing
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "warmup_steps": 0,
            "log_interval": 1,
            "save_interval": 100,
            "max_length": 128,
            "load_in_8bit": False,
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
    }
    
    # Write test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Run SFT training
        cmd = [
            sys.executable, "-m", "rlhf.sft_train",
            "--config", config_path,
            "--train-data", "data/processed/oasst1/sft_train.jsonl",
            "--valid-data", "data/processed/oasst1/sft_valid.jsonl",
            "--output", "test_output/sft"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n--- STDERR ---")
            print(result.stderr)
            print(f"\n❌ SFT training failed with exit code {result.returncode}")
            return False
        
        print("\n✓ SFT training test passed!")
        return True
        
    finally:
        os.unlink(config_path)
        if Path("test_output/sft").exists():
            shutil.rmtree("test_output/sft")


def test_rm_training():
    """Test Reward Model training script."""
    print("\n" + "="*60)
    print("TEST 2: Reward Model Training")
    print("="*60)
    
    # Create a minimal test config
    test_config = {
        "global": {"seed": 42},
        "reward_model": {
            "base_model": "facebook/opt-350m",
            "output_dir": "test_output/rm",
            "max_steps": 2,  # Just 2 steps for testing
            "batch_size": 1,
            "learning_rate": 1e-5,
            "log_interval": 1,
            "max_length": 128,
            "margin": 0.0
        }
    }
    
    # Write test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Run RM training
        cmd = [
            sys.executable, "-m", "rlhf.rm_train",
            "--config", config_path,
            "--train-data", "data/processed/preferences/pairs.jsonl",
            "--valid-data", "data/processed/preferences/pairs_valid.jsonl",
            "--output", "test_output/rm"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n--- STDERR ---")
            print(result.stderr)
            print(f"\n❌ RM training failed with exit code {result.returncode}")
            return False
        
        print("\n✓ RM training test passed!")
        return True
        
    finally:
        os.unlink(config_path)
        if Path("test_output/rm").exists():
            shutil.rmtree("test_output/rm")


def test_ppo_decision_training():
    """Test PPO decision policy training script."""
    print("\n" + "="*60)
    print("TEST 3: PPO Decision Policy Training")
    print("="*60)
    
    # Create a minimal test config
    test_config = {
        "global": {"seed": 42},
        "ppo_decision": {
            "output_dir": "test_output/ppo_decision",
            "environment": {
                "episode_length": 10,
                "initial_budget": 1000.0,
                "cooldown_days": 7,
                "fatigue_cap": 3,
                "lambda_compliance": 1.0,
                "lambda_fatigue": 1.0
            },
            "ppo": {
                "num_episodes": 2,  # Just 2 episodes for testing
                "hidden_dim": 64,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "update_epochs": 2,
                "max_grad_norm": 0.5,
                "lagrangian_step_size": 0.01
            }
        }
    }
    
    # Write test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Run PPO decision training (without models for now)
        cmd = [
            sys.executable, "-m", "agents.ppo_policy",
            "--config", config_path,
            "--output", "test_output/ppo_decision"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("\n--- STDOUT ---")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n--- STDERR ---")
            print(result.stderr)
            print(f"\n❌ PPO decision training failed with exit code {result.returncode}")
            return False
        
        print("\n✓ PPO decision training test passed!")
        return True
        
    finally:
        os.unlink(config_path)
        if Path("test_output/ppo_decision").exists():
            shutil.rmtree("test_output/ppo_decision")


def test_config_loading():
    """Test that experiment config can be loaded and parsed correctly."""
    print("\n" + "="*60)
    print("TEST 4: Experiment Config Loading")
    print("="*60)
    
    config_path = "ops/configs/experiment_exp_001_mvp.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = ["global", "sft", "reward_model", "ppo_decision"]
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
            return False
        print(f"✓ Found section: {section}")
    
    # Check SFT config
    sft = config["sft"]
    assert "base_model" in sft, "Missing sft.base_model"
    assert "lora" in sft, "Missing sft.lora"
    assert "r" in sft["lora"], "Missing sft.lora.r"
    print("✓ SFT config structure valid")
    
    # Check RM config
    rm = config["reward_model"]
    assert "base_model" in rm, "Missing reward_model.base_model"
    print("✓ RM config structure valid")
    
    # Check PPO decision config
    ppo = config["ppo_decision"]
    assert "learning_rate" in ppo, "Missing ppo_decision.learning_rate"
    assert "gamma" in ppo, "Missing ppo_decision.gamma"
    print("✓ PPO decision config structure valid")
    
    print("\n✓ Config loading test passed!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("OFFLINE TRAINING INTEGRATION TESTS")
    print("="*60)
    
    # Create test output directory
    Path("test_output").mkdir(exist_ok=True)
    
    results = {}
    
    # Test 1: Config loading (fast)
    results["config_loading"] = test_config_loading()
    
    # Test 2: PPO decision (fast, no GPU needed)
    results["ppo_decision"] = test_ppo_decision_training()
    
    # Test 3: SFT training (slow, needs GPU ideally)
    print("\n⚠️  SFT and RM tests require downloading models and are slow.")
    print("Skipping for now. Run manually if needed:")
    print("  python tests/test_training_offline.py --full")
    
    if "--full" in sys.argv:
        results["sft"] = test_sft_training()
        results["rm"] = test_rm_training()
    else:
        results["sft"] = None
        results["rm"] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{status}: {test_name}")
    
    # Cleanup
    if Path("test_output").exists():
        shutil.rmtree("test_output")
    
    # Exit code
    failed = [name for name, passed in results.items() if passed is False]
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

