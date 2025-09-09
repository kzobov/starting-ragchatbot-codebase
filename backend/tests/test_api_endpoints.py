"""
API endpoint tests for the FastAPI RAG system
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test class for API endpoint functionality"""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System API"}

    def test_query_endpoint_success(self, test_client):
        """Test successful query to /api/query endpoint"""
        query_data = {
            "query": "What is Python?",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Check types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check content
        assert data["answer"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["course"] == "Test Course"
        assert data["session_id"] == "test-session-123"

    def test_query_endpoint_with_session_id(self, test_client):
        """Test query endpoint with existing session ID"""
        query_data = {
            "query": "Tell me about variables",
            "session_id": "existing-session-456"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return the provided session ID
        assert data["session_id"] == "existing-session-456"

    def test_query_endpoint_missing_query(self, test_client):
        """Test query endpoint with missing query field"""
        query_data = {
            "session_id": "test-session"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should return validation error
        assert response.status_code == 422

    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        query_data = {
            "query": "",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should still succeed (empty query handling is up to the RAG system)
        assert response.status_code == 200

    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_courses_endpoint_success(self, test_client):
        """Test successful request to /api/courses endpoint"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Check types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check content
        assert data["total_courses"] == 2
        assert "Python Basics" in data["course_titles"]
        assert "Advanced Python" in data["course_titles"]

    def test_query_endpoint_response_format(self, test_client):
        """Test that query endpoint returns properly formatted JSON response"""
        query_data = {"query": "test"}
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Verify it's valid JSON
        data = response.json()
        assert isinstance(data, dict)

    def test_content_type_validation(self, test_client):
        """Test that endpoints validate content type properly"""
        # Test with wrong content type
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422


@pytest.mark.api
@pytest.mark.integration
class TestAPIErrorHandling:
    """Test error handling in API endpoints"""

    def test_query_endpoint_internal_error(self, test_app):
        """Test query endpoint handles internal errors gracefully"""
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from pydantic import BaseModel
        from typing import List, Optional, Dict, Any
        
        # Create app with failing RAG system
        app = FastAPI(title="Error Test App")
        
        class QueryRequest(BaseModel):
            query: str
            session_id: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Dict[str, Any]]
            session_id: str
        
        # Mock RAG system that throws errors
        class FailingRAGSystem:
            def __init__(self):
                self.session_manager = Mock()
            
            def query(self, query: str, session_id: str):
                raise ValueError("Test error")
        
        failing_rag_system = FailingRAGSystem()
        
        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            try:
                session_id = request.session_id or "test-session"
                answer, sources = failing_rag_system.query(request.query, session_id)
                return QueryResponse(
                    answer=answer,
                    sources=sources,
                    session_id=session_id
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        client = TestClient(app)
        
        query_data = {"query": "test"}
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    def test_courses_endpoint_internal_error(self, test_app):
        """Test courses endpoint handles internal errors gracefully"""
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from pydantic import BaseModel
        from typing import List
        
        # Create app with failing RAG system
        app = FastAPI(title="Error Test App")
        
        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        # Mock RAG system that throws errors
        class FailingRAGSystem:
            def get_course_analytics(self):
                raise ConnectionError("Database connection failed")
        
        failing_rag_system = FailingRAGSystem()
        
        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            try:
                analytics = failing_rag_system.get_course_analytics()
                return CourseStats(
                    total_courses=analytics["total_courses"],
                    course_titles=analytics["course_titles"]
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        client = TestClient(app)
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Test performance characteristics of API endpoints"""

    def test_concurrent_queries(self, test_client):
        """Test that API can handle concurrent requests"""
        import concurrent.futures
        import time
        
        def make_query(query_text):
            query_data = {"query": f"Test query {query_text}"}
            response = test_client.post("/api/query", json=query_data)
            return response.status_code
        
        # Make 5 concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        
        # Should complete reasonably quickly (less than 5 seconds for 5 requests)
        assert end_time - start_time < 5.0

    def test_large_query_handling(self, test_client):
        """Test API handles large query payloads"""
        large_query = "What is Python? " * 100  # Create a large query
        
        query_data = {
            "query": large_query,
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data


@pytest.mark.api
class TestAPIResponseFormats:
    """Test API response format compliance"""

    def test_query_response_schema(self, test_client):
        """Test that query response matches expected schema"""
        query_data = {"query": "test query"}
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Sources structure
        if data["sources"]:
            source = data["sources"][0]
            assert isinstance(source, dict)

    def test_courses_response_schema(self, test_client):
        """Test that courses response matches expected schema"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Non-negative count
        assert data["total_courses"] >= 0
        
        # Course titles should be strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_response_headers(self, test_client):
        """Test that responses have correct headers"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_http_methods(self, test_client):
        """Test that endpoints respond to correct HTTP methods"""
        # GET should work for courses endpoint
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        
        # POST should not work for courses endpoint
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405
        
        # GET should not work for query endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405
        
        # POST should work for query endpoint
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200