"""
Pytest configuration and shared fixtures for RAG system tests
"""

import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add parent directory to sys.path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from config import Config


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_chroma_db):
    """Create test configuration with temporary database"""
    class TestConfig(Config):
        CHROMA_PATH = temp_chroma_db
        ANTHROPIC_API_KEY = "test-api-key"
        ANTHROPIC_MODEL = "claude-3-haiku-20240307"
        CHUNK_SIZE = 500
        CHUNK_OVERLAP = 50
        MAX_RESULTS = 3
        MAX_HISTORY = 10
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    return TestConfig()


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to Python", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="http://example.com/lesson2"),
        Lesson(lesson_number=3, title="Control Structures", lesson_link="http://example.com/lesson3"),
    ]
    
    return Course(
        title="Python Basics Course",
        course_link="http://example.com/python-course",
        instructor="John Doe",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    
    # Lesson 1 chunks
    chunks.append(CourseChunk(
        content="This is an introduction to Python programming. Python is a high-level programming language.",
        course_title=sample_course.title,
        lesson_number=1,
        chunk_index=0
    ))
    
    chunks.append(CourseChunk(
        content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        course_title=sample_course.title,
        lesson_number=1,
        chunk_index=1
    ))
    
    # Lesson 2 chunks
    chunks.append(CourseChunk(
        content="Variables in Python are containers for storing data values. Python has no command for declaring a variable.",
        course_title=sample_course.title,
        lesson_number=2,
        chunk_index=2
    ))
    
    chunks.append(CourseChunk(
        content="Python has several built-in data types including integers, floats, strings, and booleans.",
        course_title=sample_course.title,
        lesson_number=2,
        chunk_index=3
    ))
    
    # Lesson 3 chunks
    chunks.append(CourseChunk(
        content="Control structures in Python include if statements, for loops, and while loops.",
        course_title=sample_course.title,
        lesson_number=3,
        chunk_index=4
    ))
    
    return chunks


@pytest.fixture
def empty_course():
    """Create an empty course for testing edge cases"""
    return Course(
        title="Empty Course",
        course_link="http://example.com/empty",
        instructor="Empty Instructor",
        lessons=[]
    )


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI interactions"""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response from Claude")]
    mock_response.stop_reason = "stop"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic response that includes tool usage"""
    mock_response = Mock()
    
    # Mock tool use content
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_123"
    mock_tool_content.input = {
        "query": "python variables",
        "course_name": "Python Basics"
    }
    
    mock_response.content = [mock_tool_content]
    mock_response.stop_reason = "tool_use"
    
    return mock_response


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.get_conversation_history.return_value = "Previous conversation context"
    mock_manager.add_exchange.return_value = None
    
    return mock_manager


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return {
        'documents': [["Python is a programming language", "Variables store data values"]],
        'metadatas': [[
            {'course_title': 'Python Basics Course', 'lesson_number': 1, 'chunk_index': 0},
            {'course_title': 'Python Basics Course', 'lesson_number': 2, 'chunk_index': 2}
        ]],
        'distances': [[0.1, 0.3]],
        'ids': [['Python_Basics_Course_0', 'Python_Basics_Course_2']]
    }


@pytest.fixture
def error_search_results():
    """Empty search results for testing error cases"""
    return {
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]],
        'ids': [[]]
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    original_env = os.environ.copy()
    os.environ['ANTHROPIC_API_KEY'] = 'test-api-key'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)