"""
End-to-end tests for RAGSystem query handling
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class TestRAGSystemQueryHandling:
    """End-to-end tests for RAG system query processing"""
    
    @pytest.fixture
    def mock_config(self, temp_chroma_db):
        """Create mock configuration for testing"""
        config = Mock(spec=Config)
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-3-haiku-20240307"
        config.CHUNK_SIZE = 500
        config.CHUNK_OVERLAP = 50
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 10
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        return config
    
    @pytest.fixture
    def rag_system_with_mocks(self, mock_config):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_generator_class, \
             patch('rag_system.SessionManager') as mock_session_manager_class:
            
            # Create RAGSystem instance
            rag_system = RAGSystem(mock_config)
            
            # Setup mocks
            rag_system.mock_vector_store = mock_vector_store_class.return_value
            rag_system.mock_ai_generator = mock_ai_generator_class.return_value
            rag_system.mock_session_manager = mock_session_manager_class.return_value
            
            return rag_system
    
    def test_successful_content_query_with_results(self, rag_system_with_mocks):
        """Test successful content-related query that returns results"""
        rag_system = rag_system_with_mocks
        
        # Mock session manager
        rag_system.mock_session_manager.get_conversation_history.return_value = "Previous context"
        
        # Mock AI generator to return response
        rag_system.mock_ai_generator.generate_response.return_value = "Python variables are containers for storing data values."
        
        # Mock tool manager to return sources
        mock_sources = [
            {
                'course_title': 'Python Basics Course',
                'lesson_number': 2,
                'lesson_link': 'http://example.com/lesson2'
            }
        ]
        rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = rag_system.query("What are Python variables?", "session-123")
        
        # Verify AI generator was called correctly
        rag_system.mock_ai_generator.generate_response.assert_called_once_with(
            query="Answer this question about course materials: What are Python variables?",
            conversation_history="Previous context",
            tools=rag_system.tool_manager.get_tool_definitions(),
            tool_manager=rag_system.tool_manager
        )
        
        # Verify session management
        rag_system.mock_session_manager.get_conversation_history.assert_called_once_with("session-123")
        rag_system.mock_session_manager.add_exchange.assert_called_once_with(
            "session-123", 
            "What are Python variables?", 
            "Python variables are containers for storing data values."
        )
        
        # Verify response and sources
        assert response == "Python variables are containers for storing data values."
        assert sources == mock_sources
        
        # Verify sources were reset
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_without_session(self, rag_system_with_mocks):
        """Test query without session ID (no history)"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator response
        rag_system.mock_ai_generator.generate_response.return_value = "General programming answer"
        
        # Mock empty sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query without session
        response, sources = rag_system.query("What is programming?")
        
        # Verify no history was requested
        rag_system.mock_session_manager.get_conversation_history.assert_not_called()
        rag_system.mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator called without history
        rag_system.mock_ai_generator.generate_response.assert_called_once_with(
            query="Answer this question about course materials: What is programming?",
            conversation_history=None,
            tools=rag_system.tool_manager.get_tool_definitions(),
            tool_manager=rag_system.tool_manager
        )
        
        assert response == "General programming answer"
        assert sources == []
    
    def test_query_with_ai_generator_error(self, rag_system_with_mocks):
        """Test handling of AI generator errors"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator to raise exception
        rag_system.mock_ai_generator.generate_response.side_effect = Exception("API rate limit exceeded")
        
        # Execute query - should propagate exception
        with pytest.raises(Exception, match="API rate limit exceeded"):
            rag_system.query("Test query", "session-123")
    
    def test_query_with_tool_execution_error(self, rag_system_with_mocks):
        """Test query when tool execution returns error"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator to return error message (tool would have failed)
        rag_system.mock_ai_generator.generate_response.return_value = "I apologize, but I couldn't search the course materials due to a technical issue."
        
        # Mock empty sources (no successful search)
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = rag_system.query("Search for something", "session-123")
        
        # Verify error response
        assert "couldn't search the course materials" in response
        assert sources == []
    
    def test_query_with_empty_database(self, rag_system_with_mocks):
        """Test query when database has no content"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator response for no results
        rag_system.mock_ai_generator.generate_response.return_value = "I couldn't find any relevant information in the course materials about that topic."
        
        # Mock empty sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = rag_system.query("Find nonexistent topic", "session-123")
        
        # Verify appropriate response
        assert "couldn't find any relevant information" in response
        assert sources == []
    
    def test_multiple_successive_queries_with_session(self, rag_system_with_mocks):
        """Test multiple queries in the same session"""
        rag_system = rag_system_with_mocks
        
        # Mock session manager to build up history
        history_responses = [
            None,  # First query - no history
            "User: What is Python?\nAssistant: Python is a programming language.",  # Second query - has history
            "User: What is Python?\nAssistant: Python is a programming language.\nUser: What are variables?\nAssistant: Variables store data."  # Third query
        ]
        rag_system.mock_session_manager.get_conversation_history.side_effect = history_responses
        
        # Mock AI responses
        ai_responses = [
            "Python is a programming language.",
            "Variables store data.",
            "Functions are reusable code blocks."
        ]
        rag_system.mock_ai_generator.generate_response.side_effect = ai_responses
        
        # Mock empty sources for simplicity
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        session_id = "session-123"
        
        # Execute first query
        response1, _ = rag_system.query("What is Python?", session_id)
        assert response1 == "Python is a programming language."
        
        # Execute second query
        response2, _ = rag_system.query("What are variables?", session_id)
        assert response2 == "Variables store data."
        
        # Execute third query
        response3, _ = rag_system.query("What are functions?", session_id)
        assert response3 == "Functions are reusable code blocks."
        
        # Verify session management was called correctly
        assert rag_system.mock_session_manager.add_exchange.call_count == 3
        assert rag_system.mock_session_manager.get_conversation_history.call_count == 3
    
    def test_tool_manager_integration(self, rag_system_with_mocks):
        """Test proper integration with tool manager"""
        rag_system = rag_system_with_mocks
        
        # Verify tool manager was set up correctly
        assert rag_system.tool_manager is not None
        
        # Verify tools were registered
        tools = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool['name'] for tool in tools]
        assert 'search_course_content' in tool_names
        assert 'get_course_outline' in tool_names
    
    def test_query_prompt_format(self, rag_system_with_mocks):
        """Test that query prompt is properly formatted"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator
        rag_system.mock_ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        rag_system.query("Test question", "session-123")
        
        # Verify prompt formatting
        call_args = rag_system.mock_ai_generator.generate_response.call_args[1]
        query_param = call_args['query']
        assert query_param == "Answer this question about course materials: Test question"
    
    def test_sources_reset_after_query(self, rag_system_with_mocks):
        """Test that sources are properly reset after each query"""
        rag_system = rag_system_with_mocks
        
        # Mock response with sources
        rag_system.mock_ai_generator.generate_response.return_value = "Response with sources"
        test_sources = [{'course_title': 'Test Course'}]
        rag_system.tool_manager.get_last_sources = Mock(return_value=test_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = rag_system.query("Test query")
        
        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
        
        # Verify sources were returned
        assert sources == test_sources


class TestRAGSystemCourseManagement:
    """Test RAGSystem course document management functionality"""
    
    @pytest.fixture
    def rag_system_with_real_vector_store(self, test_config):
        """Create RAGSystem with real VectorStore but mocked other components"""
        with patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager:
            
            rag_system = RAGSystem(test_config)
            rag_system.mock_doc_processor = mock_doc_processor.return_value
            rag_system.mock_ai_generator = mock_ai_generator.return_value
            rag_system.mock_session_manager = mock_session_manager.return_value
            
            return rag_system
    
    def test_add_course_document_success(self, rag_system_with_real_vector_store, sample_course, sample_course_chunks):
        """Test successful addition of course document"""
        rag_system = rag_system_with_real_vector_store
        
        # Mock document processor
        rag_system.mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        # Execute
        course, chunk_count = rag_system.add_course_document("/fake/path/course.txt")
        
        # Verify document processor was called
        rag_system.mock_doc_processor.process_course_document.assert_called_once_with("/fake/path/course.txt")
        
        # Verify results
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)
    
    def test_add_course_document_error(self, rag_system_with_real_vector_store):
        """Test handling of document processing error"""
        rag_system = rag_system_with_real_vector_store
        
        # Mock document processor to raise error
        rag_system.mock_doc_processor.process_course_document.side_effect = Exception("File not found")
        
        # Execute - should handle error gracefully
        course, chunk_count = rag_system.add_course_document("/fake/path/nonexistent.txt")
        
        # Verify error handling
        assert course is None
        assert chunk_count == 0
    
    def test_get_course_analytics(self, rag_system_with_real_vector_store):
        """Test course analytics retrieval"""
        rag_system = rag_system_with_real_vector_store
        
        # Add some mock data to vector store
        with patch.object(rag_system.vector_store, 'get_course_count', return_value=3), \
             patch.object(rag_system.vector_store, 'get_existing_course_titles', return_value=['Course A', 'Course B', 'Course C']):
            
            # Execute
            analytics = rag_system.get_course_analytics()
            
            # Verify results
            assert analytics['total_courses'] == 3
            assert analytics['course_titles'] == ['Course A', 'Course B', 'Course C']


class TestRAGSystemIntegrationEdgeCases:
    """Test edge cases and error scenarios in RAG system integration"""
    
    def test_malformed_query_input(self, rag_system_with_mocks):
        """Test handling of malformed query input"""
        rag_system = rag_system_with_mocks
        
        # Mock AI generator to handle malformed input
        rag_system.mock_ai_generator.generate_response.return_value = "I need more information to help you."
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Test empty query
        response, sources = rag_system.query("", "session-123")
        assert "I need more information" in response
        
        # Test very long query
        long_query = "What is " * 1000 + "Python?"
        response, sources = rag_system.query(long_query, "session-123")
        
        # Should still process (AI generator should handle truncation if needed)
        rag_system.mock_ai_generator.generate_response.assert_called()
    
    def test_session_id_edge_cases(self, rag_system_with_mocks):
        """Test handling of various session ID formats"""
        rag_system = rag_system_with_mocks
        
        rag_system.mock_ai_generator.generate_response.return_value = "Test response"
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        # Test empty string session ID
        response, _ = rag_system.query("Test", "")
        # Should not call session methods for empty string
        rag_system.mock_session_manager.get_conversation_history.assert_not_called()
        
        # Test None session ID
        rag_system.mock_session_manager.reset_mock()
        response, _ = rag_system.query("Test", None)
        # Should not call session methods for None
        rag_system.mock_session_manager.get_conversation_history.assert_not_called()
        
        # Test very long session ID
        long_session_id = "session-" + "x" * 1000
        rag_system.mock_session_manager.reset_mock()
        response, _ = rag_system.query("Test", long_session_id)
        # Should still work
        rag_system.mock_session_manager.get_conversation_history.assert_called_once_with(long_session_id)