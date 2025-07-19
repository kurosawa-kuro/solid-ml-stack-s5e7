"""
Test cases for time_tracker module - basic functionality tests.
"""

import pytest
import tempfile
import os
import time
from pathlib import Path
from src.util.time_tracker import WorkflowTimeTracker, WorkflowTimer, time_workflow


class TestWorkflowTimeTracker:
    """Test WorkflowTimeTracker basic functionality."""
    
    def test_init_new_file(self):
        """Test initialization with non-existent file."""
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tracker = WorkflowTimeTracker(tmp.name)
            assert tracker.data == {"workflows": {}}
    
    def test_start_and_end_workflow(self):
        """Test basic workflow timing."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                tracker = WorkflowTimeTracker(tmp.name)
                
                # Start workflow
                start_time = tracker.start_workflow("test_workflow")
                assert isinstance(start_time, float)
                
                # Small delay
                time.sleep(0.1)
                
                # End workflow
                tracker.end_workflow("test_workflow", start_time)
                
                # Check data was saved
                assert "test_workflow" in tracker.data["workflows"]
                executions = tracker.data["workflows"]["test_workflow"]["executions"]
                assert len(executions) == 1
                assert executions[0]["duration"] >= 0.1
                
            finally:
                os.unlink(tmp.name)
    
    def test_get_estimated_duration(self):
        """Test duration estimation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                tracker = WorkflowTimeTracker(tmp.name)
                
                # No estimate for new workflow
                assert tracker.get_estimated_duration("new_workflow") is None
                
                # Add some executions
                start_time = tracker.start_workflow("test_workflow")
                time.sleep(0.1)
                tracker.end_workflow("test_workflow", start_time)
                
                # Should have estimate now
                estimate = tracker.get_estimated_duration("test_workflow")
                assert estimate is not None
                assert estimate >= 0.1
                
            finally:
                os.unlink(tmp.name)
    
    def test_workflow_stats(self):
        """Test statistics generation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                tracker = WorkflowTimeTracker(tmp.name)
                
                # No stats for non-existent workflow
                assert tracker.get_workflow_stats("missing") is None
                
                # Add execution
                start_time = tracker.start_workflow("test_workflow")
                time.sleep(0.1)
                tracker.end_workflow("test_workflow", start_time)
                
                # Check stats
                stats = tracker.get_workflow_stats("test_workflow")
                assert stats is not None
                assert "average" in stats
                assert "median" in stats
                assert "count" in stats
                assert stats["count"] == 1
                
            finally:
                os.unlink(tmp.name)


class TestWorkflowTimer:
    """Test WorkflowTimer context manager."""
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                tracker = WorkflowTimeTracker(tmp.name)
                
                # Use context manager
                with WorkflowTimer(tracker, "context_test"):
                    time.sleep(0.1)
                
                # Check workflow was recorded
                assert "context_test" in tracker.data["workflows"]
                executions = tracker.data["workflows"]["context_test"]["executions"]
                assert len(executions) == 1
                assert executions[0]["duration"] >= 0.1
                
            finally:
                os.unlink(tmp.name)


class TestTimeWorkflowDecorator:
    """Test time_workflow decorator."""
    
    def test_decorator(self):
        """Test decorator functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                @time_workflow("decorator_test", tmp.name)
                def test_function():
                    time.sleep(0.1)
                    return "success"
                
                # Call decorated function
                result = test_function()
                assert result == "success"
                
                # Check workflow was recorded
                tracker = WorkflowTimeTracker(tmp.name)
                assert "decorator_test" in tracker.data["workflows"]
                executions = tracker.data["workflows"]["decorator_test"]["executions"]
                assert len(executions) == 1
                assert executions[0]["duration"] >= 0.1
                
            finally:
                os.unlink(tmp.name)


class TestDataPersistence:
    """Test data persistence and file operations."""
    
    def test_data_persistence(self):
        """Test that data persists across instances."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # First instance
                tracker1 = WorkflowTimeTracker(tmp.name)
                start_time = tracker1.start_workflow("persist_test")
                time.sleep(0.1)
                tracker1.end_workflow("persist_test", start_time)
                
                # Second instance should load existing data
                tracker2 = WorkflowTimeTracker(tmp.name)
                assert "persist_test" in tracker2.data["workflows"]
                
                # Add another execution
                start_time = tracker2.start_workflow("persist_test")
                time.sleep(0.1)
                tracker2.end_workflow("persist_test", start_time)
                
                # Should have 2 executions
                executions = tracker2.data["workflows"]["persist_test"]["executions"]
                assert len(executions) == 2
                
            finally:
                os.unlink(tmp.name)
    
    def test_directory_creation(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            db_path = os.path.join(tmpdir, "nested", "dir", "tracker.json")
            tracker = WorkflowTimeTracker(db_path)
            
            # Add workflow to trigger save
            start_time = tracker.start_workflow("dir_test")
            tracker.end_workflow("dir_test", start_time)
            
            # Check file was created
            assert os.path.exists(db_path)