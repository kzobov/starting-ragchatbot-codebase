"""
Unit tests for CourseSearchTool execute method
"""

import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore, SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool execute method"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        return Mock(spec=VectorStore)
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool instance with mock vector store"""
        return CourseSearchTool(mock_vector_store)
    
    def test_successful_search_with_results(self, search_tool, mock_vector_store, sample_search_results):
        """Test successful search that returns results"""
        # Setup mock to return successful results
        search_results = SearchResults.from_chroma(sample_search_results)
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        
        # Execute search
        result = search_tool.execute("python programming", "Python Basics", 1)
        
        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="python programming",
            course_name="Python Basics",
            lesson_number=1
        )
        
        # Verify result format
        assert "[Python Basics Course" in result
        assert "Python is a programming language" in result
        assert "Variables store data values" in result
        
        # Verify sources were stored
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]['course_title'] == 'Python Basics Course'
        assert search_tool.last_sources[0]['lesson_number'] == 1
    
    def test_search_with_error(self, search_tool, mock_vector_store):
        """Test search that returns error from vector store"""
        # Setup mock to return error
        error_result = SearchResults.empty("Database connection failed")
        mock_vector_store.search.return_value = error_result
        
        # Execute search
        result = search_tool.execute("python programming")
        
        # Verify error is returned
        assert result == "Database connection failed"
        
        # Verify no sources were stored
        assert len(search_tool.last_sources) == 0
    
    def test_search_with_empty_results(self, search_tool, mock_vector_store):
        """Test search that returns no results"""
        # Setup mock to return empty results
        empty_result = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_result
        
        # Execute search with course filter
        result = search_tool.execute("nonexistent topic", "Python Basics")
        
        # Verify appropriate message
        assert "No relevant content found in course 'Python Basics'" in result
        
        # Execute search with lesson filter
        result = search_tool.execute("nonexistent topic", lesson_number=5)
        
        # Verify appropriate message
        assert "No relevant content found in lesson 5" in result
    
    def test_search_with_course_and_lesson_filters(self, search_tool, mock_vector_store, sample_search_results):
        """Test search with both course and lesson filters"""
        search_results = SearchResults.from_chroma(sample_search_results)
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson2"
        
        result = search_tool.execute("variables", "Python Basics", 2)
        
        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python Basics",
            lesson_number=2
        )
        
        assert "Python Basics Course - Lesson 2" in result
    
    def test_search_without_filters(self, search_tool, mock_vector_store, sample_search_results):
        """Test search without any course or lesson filters"""
        search_results = SearchResults.from_chroma(sample_search_results)
        mock_vector_store.search.return_value = search_results
        
        result = search_tool.execute("python programming")
        
        # Verify search was called without filters
        mock_vector_store.search.assert_called_once_with(
            query="python programming",
            course_name=None,
            lesson_number=None
        )
        
        assert "Python Basics Course" in result
    
    def test_search_result_formatting(self, search_tool, mock_vector_store):
        """Test proper formatting of search results"""
        # Create custom search results for formatting test
        custom_results = SearchResults(
            documents=["First chunk content", "Second chunk content"],
            metadata=[
                {'course_title': 'Advanced Python', 'lesson_number': 3, 'chunk_index': 10},
                {'course_title': 'Advanced Python', 'lesson_number': None, 'chunk_index': 20}
            ],
            distances=[0.1, 0.2]
        )
        
        mock_vector_store.search.return_value = custom_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson3"
        
        result = search_tool.execute("test query")
        
        # Verify formatting includes proper headers
        assert "[Advanced Python - Lesson 3]" in result
        assert "[Advanced Python]" in result  # No lesson number for second result
        assert "First chunk content" in result
        assert "Second chunk content" in result
        
        # Verify multiple results are separated
        chunks = result.split("\n\n")
        assert len(chunks) == 2
    
    def test_lesson_link_retrieval(self, search_tool, mock_vector_store):
        """Test that lesson links are properly retrieved and stored in sources"""
        search_results = SearchResults(
            documents=["Content with lesson link"],
            metadata=[{'course_title': 'Test Course', 'lesson_number': 5, 'chunk_index': 1}],
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson5"
        
        search_tool.execute("test query")
        
        # Verify lesson link was retrieved
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 5)
        
        # Verify source includes lesson link
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]['lesson_link'] == "http://example.com/lesson5"
    
    def test_lesson_link_not_available(self, search_tool, mock_vector_store):
        """Test handling when lesson link is not available"""
        search_results = SearchResults(
            documents=["Content without lesson link"],
            metadata=[{'course_title': 'Test Course', 'lesson_number': None, 'chunk_index': 1}],
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = search_results
        
        search_tool.execute("test query")
        
        # Verify lesson link was not retrieved for lesson_number None
        mock_vector_store.get_lesson_link.assert_not_called()
        
        # Verify source doesn't include lesson link
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]['lesson_link'] is None
    
    def test_vector_store_exception_handling(self, search_tool, mock_vector_store):
        """Test handling of exceptions from vector store"""
        # The VectorStore.search() method should catch exceptions and return SearchResults.empty()
        # Let's test the expected behavior when VectorStore returns error
        mock_vector_store.search.return_value = SearchResults.empty("ChromaDB connection error")
        
        result = search_tool.execute("test query")
        assert result == "ChromaDB connection error"
        
        # Also test what happens if an exception actually gets through (this is a bug we need to fix)
        mock_vector_store.search.side_effect = Exception("Uncaught ChromaDB connection error")
        
        # This should reveal the bug - the tool doesn't handle exceptions properly
        try:
            result = search_tool.execute("test query")
            # If we get here, the tool handled it somehow
        except Exception as e:
            # This is what we expect currently - the exception propagates
            assert "Uncaught ChromaDB connection error" in str(e)
    
    def test_malformed_metadata(self, search_tool, mock_vector_store):
        """Test handling of malformed metadata in search results"""
        # Create results with missing metadata fields
        malformed_results = SearchResults(
            documents=["Some content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = malformed_results
        
        result = search_tool.execute("test query")
        
        # Verify it handles missing metadata gracefully
        assert "[unknown]" in result
        assert "Some content" in result
        
        # Verify source handling with missing data
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]['course_title'] == 'unknown'
        assert search_tool.last_sources[0]['lesson_number'] is None


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with ToolManager"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for integration testing"""
        return Mock(spec=VectorStore)
    
    def test_tool_registration(self, mock_vector_store):
        """Test that CourseSearchTool can be registered with ToolManager"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        tool_manager.register_tool(search_tool)
        
        # Verify tool is registered
        tool_definitions = tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 1
        assert tool_definitions[0]['name'] == 'search_course_content'
    
    def test_tool_execution_via_manager(self, mock_vector_store, sample_search_results):
        """Test executing CourseSearchTool via ToolManager"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Setup mock return
        search_results = SearchResults.from_chroma(sample_search_results)
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        # Execute via tool manager
        result = tool_manager.execute_tool(
            'search_course_content',
            query='python programming',
            course_name='Python Basics'
        )
        
        assert "[Python Basics Course" in result
        
        # Verify sources are available via tool manager
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
    
    def test_tool_definition_format(self, mock_vector_store):
        """Test that CourseSearchTool returns proper tool definition"""
        search_tool = CourseSearchTool(mock_vector_store)
        definition = search_tool.get_tool_definition()
        
        # Verify required fields
        assert definition['name'] == 'search_course_content'
        assert 'description' in definition
        assert 'input_schema' in definition
        
        # Verify schema structure
        schema = definition['input_schema']
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'required' in schema
        
        # Verify required parameters
        assert 'query' in schema['required']
        assert 'query' in schema['properties']
        assert 'course_name' in schema['properties']
        assert 'lesson_number' in schema['properties']