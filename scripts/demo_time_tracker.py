#!/usr/bin/env python3
"""
Demo script showing time tracker integration with ML workflows.
"""

import os
import sys
import time
from pathlib import Path

from util.time_tracker import WorkflowTimer, WorkflowTimeTracker, time_workflow

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def demo_ml_workflow():
    """Demonstrate time tracking in a simulated ML workflow."""
    print("🤖 ML Workflow Time Tracking Demo")
    print("=" * 40)

    # Initialize tracker
    tracker = WorkflowTimeTracker("data/workflow_times.json")

    # Simulate data loading
    with WorkflowTimer(tracker, "data_loading"):
        print("📊 Loading data from DuckDB...")
        time.sleep(0.5)  # Simulate data loading

    # Simulate preprocessing
    with WorkflowTimer(tracker, "preprocessing"):
        print("🔧 Preprocessing data...")
        time.sleep(1.0)  # Simulate preprocessing

    # Simulate feature engineering
    with WorkflowTimer(tracker, "feature_engineering"):
        print("🎯 Creating features...")
        time.sleep(1.5)  # Simulate feature engineering

    # Simulate model training (using decorator)
    @time_workflow("model_training", "data/workflow_times.json")
    def train_models():
        print("🧠 Training models...")
        time.sleep(2.0)  # Simulate model training
        return "Models trained successfully"

    result = train_models()
    print(f"Result: {result}")

    # Show all statistics
    print("\n📈 Workflow Performance Summary:")
    tracker.print_all_stats()

    # Demonstrate prediction for next run
    print("\n🔮 Next Run Predictions:")
    for workflow in tracker.list_workflows():
        estimated = tracker.get_estimated_duration(workflow)
        if estimated:
            print(f"  {workflow}: ~{estimated: .1f}s")


def demo_multiple_runs():
    """Demonstrate how predictions improve with multiple runs."""
    print("\n\n🔄 Multiple Runs Demo (showing prediction improvement)")
    print("=" * 60)

    tracker = WorkflowTimeTracker("data/workflow_times.json")

    # Run the same workflow multiple times with slight variations
    for i in range(3):
        print(f"\n--- Run {i + 1} ---")
        with WorkflowTimer(tracker, "data_processing"):
            # Simulate variable processing time
            processing_time = 1.0 + (i * 0.3)  # 1.0s, 1.3s, 1.6s
            time.sleep(processing_time)

        # Show how prediction becomes more accurate
        stats = tracker.get_workflow_stats("data_processing")
        if stats and stats.get("count", 0) > 1:
            estimated = tracker.get_estimated_duration("data_processing")
            print(f"   Current average: {stats['average']: .2f}s")
            print(f"   Next run estimate: {estimated: .2f}s")


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    try:
        demo_ml_workflow()
        demo_multiple_runs()

        print("\n✨ Demo completed! Check 'data/workflow_times.json' for stored data.")

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
