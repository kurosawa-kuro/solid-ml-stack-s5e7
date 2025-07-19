#!/usr/bin/env python3
"""
Command-line interface for the workflow time tracker.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from util.time_tracker import WorkflowTimeTracker


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Workflow Time Tracker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/time_tracker_cli.py --stats              # Show all statistics
  python3 scripts/time_tracker_cli.py --list               # List workflows
  python3 scripts/time_tracker_cli.py --estimate data_processing  # Get estimate
  python3 scripts/time_tracker_cli.py --clear data_processing     # Clear workflow data
        """
    )
    
    parser.add_argument(
        "--db-path", 
        default="data/workflow_times.json",
        help="Path to the JSON database file (default: data/workflow_times.json)"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show statistics for all workflows"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all tracked workflows"
    )
    
    parser.add_argument(
        "--estimate",
        metavar="WORKFLOW",
        help="Get estimated duration for a workflow"
    )
    
    parser.add_argument(
        "--clear",
        metavar="WORKFLOW", 
        help="Clear data for a specific workflow"
    )
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = WorkflowTimeTracker(args.db_path)
    
    if args.stats:
        tracker.print_all_stats()
        
    elif args.list:
        workflows = tracker.list_workflows()
        if workflows:
            print("📋 Tracked Workflows:")
            for workflow in workflows:
                stats = tracker.get_workflow_stats(workflow)
                count = stats.get('count', 0) if stats else 0
                print(f"  • {workflow} ({count} runs)")
        else:
            print("No workflows tracked yet.")
            
    elif args.estimate:
        estimated = tracker.get_estimated_duration(args.estimate)
        if estimated:
            print(f"⏱️  Estimated duration for '{args.estimate}': {estimated:.2f}s")
        else:
            print(f"❌ No data found for workflow '{args.estimate}'")
            
    elif args.clear:
        if tracker.clear_workflow_data(args.clear):
            print(f"✅ Cleared data for '{args.clear}'")
        else:
            print(f"❌ No data found for workflow '{args.clear}'")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()