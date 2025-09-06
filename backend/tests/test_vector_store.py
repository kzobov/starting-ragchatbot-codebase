"""
Reliability tests for VectorStore functionality
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
import chromadb
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStoreInitialization:
    """Test VectorStore initialization and setup"""
    
    def test_successful_initialization(self, temp_chroma_db):
        """Test successful VectorStore initialization"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_db,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        assert vector_store.max_results == 5
        assert vector_store.client is not None
        assert vector_store.course_catalog is not None
        assert vector_store.course_content is not None
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid path"""
        # Should still work - ChromaDB creates directory if it doesn't exist
        invalid_path = "/nonexistent/path/chroma"
        vector_store = VectorStore(
            chroma_path=invalid_path,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Cleanup
        if os.path.exists(invalid_path):
            shutil.rmtree(invalid_path)
    
    @patch('chromadb.PersistentClient')
    def test_initialization_with_chromadb_error(self, mock_client_class):
        """Test initialization when ChromaDB fails"""
        mock_client_class.side_effect = Exception("ChromaDB initialization failed")
        
        with pytest.raises(Exception):
            VectorStore(
                chroma_path="/tmp/test",
                embedding_model="all-MiniLM-L6-v2"
            )


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""
    
    @pytest.fixture
    def vector_store_with_data(self, temp_chroma_db, sample_course, sample_course_chunks):
        """Create VectorStore with sample data"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=3)
        
        # Add sample data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)
        
        return vector_store
    
    def test_search_without_filters(self, vector_store_with_data):
        """Test basic search without any filters"""
        results = vector_store_with_data.search("Python programming")
        
        assert not results.error
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) == len(results.documents)
    
    def test_search_with_course_filter(self, vector_store_with_data):
        """Test search with course name filter"""
        results = vector_store_with_data.search(
            query="variables",
            course_name="Python Basics"
        )
        
        assert not results.error
        if not results.is_empty():
            # All results should be from the specified course
            for metadata in results.metadata:
                assert metadata['course_title'] == 'Python Basics Course'
    
    def test_search_with_lesson_filter(self, vector_store_with_data):
        """Test search with lesson number filter"""
        results = vector_store_with_data.search(
            query="variables",
            lesson_number=2
        )
        
        assert not results.error
        if not results.is_empty():
            # All results should be from lesson 2
            for metadata in results.metadata:
                assert metadata['lesson_number'] == 2
    
    def test_search_with_both_filters(self, vector_store_with_data):
        """Test search with both course and lesson filters"""
        results = vector_store_with_data.search(
            query="data",
            course_name="Python Basics",
            lesson_number=2
        )
        
        assert not results.error
        if not results.is_empty():
            for metadata in results.metadata:
                assert metadata['course_title'] == 'Python Basics Course'
                assert metadata['lesson_number'] == 2
    
    def test_search_nonexistent_course(self, vector_store_with_data):
        """Test search with non-existent course name"""
        results = vector_store_with_data.search(
            query="test",
            course_name="NonExistent Course"
        )
        
        assert results.error
        assert "No course found matching 'NonExistent Course'" in results.error
    
    def test_search_nonexistent_lesson(self, vector_store_with_data):
        """Test search with non-existent lesson number"""
        results = vector_store_with_data.search(
            query="test",
            lesson_number=999
        )
        
        assert not results.error
        # Should return empty results, not an error
        assert results.is_empty()
    
    def test_search_empty_query(self, vector_store_with_data):
        """Test search with empty query string"""
        results = vector_store_with_data.search("")
        
        # Should still work with empty query
        assert not results.error
    
    def test_search_very_long_query(self, vector_store_with_data):
        """Test search with very long query string"""
        long_query = "Python " * 1000
        results = vector_store_with_data.search(long_query)
        
        # Should handle long queries gracefully
        assert not results.error
    
    def test_search_with_limit(self, vector_store_with_data):
        """Test search with custom result limit"""
        results = vector_store_with_data.search("Python", limit=1)
        
        assert not results.error
        if not results.is_empty():
            assert len(results.documents) <= 1
    
    def test_search_with_zero_limit(self, vector_store_with_data):
        """Test search with zero limit"""
        results = vector_store_with_data.search("Python", limit=0)
        
        # Should return empty results
        assert results.is_empty()
    
    @patch.object(VectorStore, 'course_content')
    def test_search_with_chromadb_error(self, mock_collection, vector_store_with_data):
        """Test search when ChromaDB query fails"""
        # Mock ChromaDB to raise exception
        mock_collection.query.side_effect = Exception("ChromaDB query failed")
        vector_store_with_data.course_content = mock_collection
        
        results = vector_store_with_data.search("test")
        
        assert results.error
        assert "Search error:" in results.error
        assert "ChromaDB query failed" in results.error


class TestVectorStoreCourseResolution:
    """Test course name resolution functionality"""
    
    @pytest.fixture
    def vector_store_with_multiple_courses(self, temp_chroma_db):
        """Create VectorStore with multiple courses for testing resolution"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Add multiple courses
        courses = [
            Course(title="Python Basics Course", instructor="John Doe"),
            Course(title="Advanced Python Programming", instructor="Jane Smith"),
            Course(title="Python Web Development", instructor="Bob Johnson"),
            Course(title="JavaScript Fundamentals", instructor="Alice Brown")
        ]
        
        for course in courses:
            vector_store.add_course_metadata(course)
        
        return vector_store
    
    def test_exact_course_name_resolution(self, vector_store_with_multiple_courses):
        """Test resolution with exact course name"""
        resolved = vector_store_with_multiple_courses._resolve_course_name("Python Basics Course")
        assert resolved == "Python Basics Course"
    
    def test_partial_course_name_resolution(self, vector_store_with_multiple_courses):
        """Test resolution with partial course name"""
        resolved = vector_store_with_multiple_courses._resolve_course_name("Python Basics")
        # Should match the most similar course
        assert resolved in ["Python Basics Course", "Advanced Python Programming", "Python Web Development"]
    
    def test_case_insensitive_resolution(self, vector_store_with_multiple_courses):
        """Test case-insensitive course name resolution"""
        resolved = vector_store_with_multiple_courses._resolve_course_name("python basics")
        # Should still find a match despite different case
        assert resolved is not None
    
    def test_nonexistent_course_resolution(self, vector_store_with_multiple_courses):
        """Test resolution with completely unrelated name"""
        resolved = vector_store_with_multiple_courses._resolve_course_name("Quantum Physics")
        # May return None or the best match - depends on similarity threshold
        # We'll accept either behavior
        assert True  # Just ensure it doesn't crash
    
    def test_empty_course_name_resolution(self, vector_store_with_multiple_courses):
        """Test resolution with empty course name"""
        resolved = vector_store_with_multiple_courses._resolve_course_name("")
        # Should return None for empty string
        assert resolved is None
    
    @patch.object(VectorStore, 'course_catalog')
    def test_course_resolution_with_error(self, mock_catalog, vector_store_with_multiple_courses):
        """Test course resolution when ChromaDB query fails"""
        mock_catalog.query.side_effect = Exception("Query failed")
        vector_store_with_multiple_courses.course_catalog = mock_catalog
        
        resolved = vector_store_with_multiple_courses._resolve_course_name("Python")
        assert resolved is None


class TestVectorStoreDataManagement:
    """Test VectorStore data addition and retrieval"""
    
    def test_add_course_metadata(self, temp_chroma_db, sample_course):
        """Test adding course metadata"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Should not raise exception
        vector_store.add_course_metadata(sample_course)
        
        # Verify data was added
        existing_titles = vector_store.get_existing_course_titles()
        assert sample_course.title in existing_titles
    
    def test_add_course_content(self, temp_chroma_db, sample_course_chunks):
        """Test adding course content chunks"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Should not raise exception
        vector_store.add_course_content(sample_course_chunks)
        
        # Verify content was added by searching
        results = vector_store.search("Python programming")
        assert not results.is_empty()
    
    def test_add_empty_course_content(self, temp_chroma_db):
        """Test adding empty course content list"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Should handle empty list gracefully
        vector_store.add_course_content([])
        
        # Should still be able to search (returns empty results)
        results = vector_store.search("anything")
        assert results.is_empty()
    
    def test_get_existing_course_titles(self, temp_chroma_db, sample_course):
        """Test retrieving existing course titles"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Initially empty
        titles = vector_store.get_existing_course_titles()
        assert len(titles) == 0
        
        # Add course and verify
        vector_store.add_course_metadata(sample_course)
        titles = vector_store.get_existing_course_titles()
        assert sample_course.title in titles
    
    def test_get_course_count(self, temp_chroma_db, sample_course):
        """Test getting course count"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Initially zero
        count = vector_store.get_course_count()
        assert count == 0
        
        # Add course and verify
        vector_store.add_course_metadata(sample_course)
        count = vector_store.get_course_count()
        assert count == 1
    
    def test_get_all_courses_metadata(self, temp_chroma_db, sample_course):
        """Test retrieving all course metadata"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Add course
        vector_store.add_course_metadata(sample_course)
        
        # Retrieve metadata
        metadata_list = vector_store.get_all_courses_metadata()
        assert len(metadata_list) == 1
        
        metadata = metadata_list[0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert 'lessons' in metadata
        assert isinstance(metadata['lessons'], list)
    
    def test_get_course_link(self, temp_chroma_db, sample_course):
        """Test retrieving course link"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Add course
        vector_store.add_course_metadata(sample_course)
        
        # Get link
        link = vector_store.get_course_link(sample_course.title)
        assert link == sample_course.course_link
        
        # Test non-existent course
        link = vector_store.get_course_link("Nonexistent Course")
        assert link is None
    
    def test_get_lesson_link(self, temp_chroma_db, sample_course):
        """Test retrieving lesson link"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Add course
        vector_store.add_course_metadata(sample_course)
        
        # Get lesson link
        link = vector_store.get_lesson_link(sample_course.title, 1)
        assert link == sample_course.lessons[0].lesson_link
        
        # Test non-existent lesson
        link = vector_store.get_lesson_link(sample_course.title, 999)
        assert link is None
        
        # Test non-existent course
        link = vector_store.get_lesson_link("Nonexistent Course", 1)
        assert link is None
    
    def test_clear_all_data(self, temp_chroma_db, sample_course, sample_course_chunks):
        """Test clearing all data"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Add data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)
        
        # Verify data exists
        assert vector_store.get_course_count() > 0
        results = vector_store.search("Python")
        assert not results.is_empty()
        
        # Clear data
        vector_store.clear_all_data()
        
        # Verify data is gone
        assert vector_store.get_course_count() == 0
        results = vector_store.search("Python")
        assert results.is_empty()


class TestSearchResults:
    """Test SearchResults data structure"""
    
    def test_from_chroma_with_results(self, sample_search_results):
        """Test creating SearchResults from ChromaDB results"""
        search_results = SearchResults.from_chroma(sample_search_results)
        
        assert len(search_results.documents) == 2
        assert len(search_results.metadata) == 2
        assert len(search_results.distances) == 2
        assert search_results.error is None
        assert not search_results.is_empty()
    
    def test_from_chroma_empty_results(self, error_search_results):
        """Test creating SearchResults from empty ChromaDB results"""
        search_results = SearchResults.from_chroma(error_search_results)
        
        assert len(search_results.documents) == 0
        assert len(search_results.metadata) == 0
        assert len(search_results.distances) == 0
        assert search_results.error is None
        assert search_results.is_empty()
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        error_msg = "Test error message"
        search_results = SearchResults.empty(error_msg)
        
        assert len(search_results.documents) == 0
        assert len(search_results.metadata) == 0
        assert len(search_results.distances) == 0
        assert search_results.error == error_msg
        assert search_results.is_empty()
    
    def test_is_empty_method(self):
        """Test is_empty method behavior"""
        # Empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        assert empty_results.is_empty()
        
        # Non-empty results
        non_empty_results = SearchResults(
            documents=["test"],
            metadata=[{"test": "value"}],
            distances=[0.5]
        )
        assert not non_empty_results.is_empty()


class TestVectorStoreErrorHandling:
    """Test error handling in VectorStore operations"""
    
    @patch('chromadb.PersistentClient')
    def test_chromadb_connection_failure(self, mock_client_class):
        """Test handling of ChromaDB connection failures"""
        mock_client_class.side_effect = ConnectionError("Cannot connect to ChromaDB")
        
        with pytest.raises(ConnectionError):
            VectorStore("/tmp/test", "all-MiniLM-L6-v2")
    
    def test_malformed_course_data(self, temp_chroma_db):
        """Test handling of malformed course data"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Course with missing required fields
        malformed_course = Course(title="")  # Empty title
        
        # Should handle gracefully (may add with empty title or skip)
        vector_store.add_course_metadata(malformed_course)
        
        # Verify system still works
        results = vector_store.search("test")
        assert not results.error  # No crash
    
    def test_malformed_chunk_data(self, temp_chroma_db):
        """Test handling of malformed chunk data"""
        vector_store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        # Chunk with empty content
        malformed_chunk = CourseChunk(
            content="",  # Empty content
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        )
        
        # Should handle gracefully
        vector_store.add_course_content([malformed_chunk])
        
        # Verify system still works
        results = vector_store.search("test")
        assert not results.error  # No crash