"""
Time Tracker Utility - JSON-based workflow execution time tracking.

This module provides functionality to track workflow execution times,
store them in JSON format, and predict completion times for future runs.
"""

import json
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class WorkflowTimeTracker:
    """JSON-based time tracker for workflow execution times."""

    def __init__(self, db_path: str = "data/workflow_times.json"):
        """Initialize the time tracker.

        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = Path(db_path)
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load existing data from JSON file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"workflows": {}}
        return {"workflows": {}}

    def _save_data(self) -> None:
        """Save data to JSON file."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def start_workflow(self, workflow_name: str) -> float:
        """Start tracking a workflow execution.

        Args:
            workflow_name: Name of the workflow to track

        Returns:
            Start time timestamp
        """
        start_time = time.time()

        # Get estimated completion time
        estimated_duration = self.get_estimated_duration(workflow_name)

        if estimated_duration:
            estimated_end = datetime.now() + timedelta(seconds=estimated_duration)
            print(f"🚀 Starting workflow: {workflow_name}")
            print(f"⏱️  Estimated completion: {estimated_end.strftime('%H:%M: %S')} " f"({int(estimated_duration)}s)")
        else:
            print(f"🚀 Starting workflow: {workflow_name} " "(first run, no estimate available)")

        return start_time

    def end_workflow(self, workflow_name: str, start_time: float) -> None:
        """End tracking and save the execution time.

        Args:
            workflow_name: Name of the workflow
            start_time: Start time timestamp from start_workflow()
        """
        duration = time.time() - start_time

        # Initialize workflow data if not exists
        if workflow_name not in self.data["workflows"]:
            self.data["workflows"][workflow_name] = {
                "executions": [],
                "statistics": {},
            }

        # Add new execution record
        execution = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
        }

        self.data["workflows"][workflow_name]["executions"].append(execution)

        # Keep only last 100 executions to prevent file bloat
        self.data["workflows"][workflow_name]["executions"] = self.data["workflows"][workflow_name]["executions"][-100:]

        # Update statistics
        self._update_statistics(workflow_name)

        # Save to file
        self._save_data()

        print(f"✅ Workflow completed in {duration: .2f}s")

    def _update_statistics(self, workflow_name: str) -> None:
        """Update statistics for a workflow.

        Args:
            workflow_name: Name of the workflow
        """
        executions = self.data["workflows"][workflow_name]["executions"]
        durations = [e["duration"] for e in executions]

        if durations:
            stats = {
                "average": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "count": len(durations),
                "last_updated": datetime.now().isoformat(),
            }

            # Add standard deviation if we have enough samples
            if len(durations) >= 2:
                stats["std_dev"] = statistics.stdev(durations)

            self.data["workflows"][workflow_name]["statistics"] = stats

    def get_estimated_duration(self, workflow_name: str) -> Optional[float]:
        """Get estimated duration based on historical data.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Estimated duration in seconds, or None if no data
        """
        if workflow_name not in self.data["workflows"]:
            return None

        stats = self.data["workflows"][workflow_name].get("statistics", {})

        if not stats:
            return None

        # Use weighted average of mean and median for more stable estimates
        if "median" in stats and "average" in stats:
            # Give more weight to median as it's less affected by outliers
            return stats["median"] * 0.7 + stats["average"] * 0.3

        return stats.get("average")

    def get_workflow_stats(self, workflow_name: str) -> Optional[Dict]:
        """Get statistics for a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Statistics dictionary or None if not found
        """
        if workflow_name in self.data["workflows"]:
            return self.data["workflows"][workflow_name]["statistics"]
        return None

    def list_workflows(self) -> List[str]:
        """List all tracked workflows.

        Returns:
            List of workflow names
        """
        return list(self.data["workflows"].keys())

    def print_all_stats(self) -> None:
        """Print statistics for all workflows."""
        if not self.data["workflows"]:
            print("No workflows tracked yet.")
            return

        print("\n📊 Workflow Statistics Summary")
        print("=" * 50)

        for workflow_name in self.list_workflows():
            stats = self.get_workflow_stats(workflow_name)
            if stats:
                print(f"\n{workflow_name}:")
                print(f"  Average: {stats.get('average', 0): .2f}s")
                print(f"  Median: {stats.get('median', 0): .2f}s")
                print(f"  Range: {stats.get('min', 0): .2f}s - " f"{stats.get('max', 0): .2f}s")
                print(f"  Runs: {stats.get('count', 0)}")
                if "std_dev" in stats:
                    print(f"  Std Dev: {stats.get('std_dev', 0): .2f}s")

    def clear_workflow_data(self, workflow_name: str) -> bool:
        """Clear data for a specific workflow.

        Args:
            workflow_name: Name of the workflow to clear

        Returns:
            True if data was cleared, False if workflow not found
        """
        if workflow_name in self.data["workflows"]:
            del self.data["workflows"][workflow_name]
            self._save_data()
            print(f"🗑️  Cleared data for workflow: {workflow_name}")
            return True
        return False


class WorkflowTimer:
    """Context manager for easy workflow timing."""

    def __init__(self, tracker: WorkflowTimeTracker, workflow_name: str):
        """Initialize the timer.

        Args:
            tracker: WorkflowTimeTracker instance
            workflow_name: Name of the workflow
        """
        self.tracker = tracker
        self.workflow_name = workflow_name
        self.start_time = None

    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = self.tracker.start_workflow(self.workflow_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing when exiting context."""
        if self.start_time:
            self.tracker.end_workflow(self.workflow_name, self.start_time)


def time_workflow(workflow_name: str, db_path: str = "data/workflow_times.json"):
    """Decorator for timing functions.

    Args:
        workflow_name: Name of the workflow
        db_path: Path to the JSON database

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = WorkflowTimeTracker(db_path)
            with WorkflowTimer(tracker, workflow_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Create a tracker instance
    tracker = WorkflowTimeTracker("test_workflow_times.json")

    # Example 1: Using context manager
    print("Example 1: Context manager usage")
    with WorkflowTimer(tracker, "data_processing"):
        time.sleep(2)  # Simulate work

    # Example 2: Manual tracking
    print("\nExample 2: Manual tracking")
    start = tracker.start_workflow("model_training")
    time.sleep(1)  # Simulate work
    tracker.end_workflow("model_training", start)

    # Example 3: Decorator usage
    print("\nExample 3: Decorator usage")

    @time_workflow("feature_engineering")
    def create_features():
        time.sleep(0.5)  # Simulate work
        return "features created"

    result = create_features()
    print(f"Result: {result}")

    # Show statistics
    print("\nAll statistics:")
    tracker.print_all_stats()

    # Clean up test file
    if os.path.exists("test_workflow_times.json"):
        os.remove("test_workflow_times.json")
        print("\n🧹 Cleaned up test file")
