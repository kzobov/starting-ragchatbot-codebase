"""
Integration tests for AIGenerator tool calling functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorToolCalling:
    """Test cases for AIGenerator tool calling functionality"""
    
    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance with test API key"""
        return AIGenerator(api_key="test-api-key", model="claude-3-haiku-20240307")
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager for testing"""
        tool_manager = Mock(spec=ToolManager)
        tool_manager.get_tool_definitions.return_value = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }]
        tool_manager.execute_tool.return_value = "Mock search results"
        return tool_manager
    
    @patch('anthropic.Anthropic')
    def test_successful_tool_execution(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test successful tool execution flow"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock initial tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {
            "query": "python variables",
            "course_name": "Python Basics"
        }
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Based on the search results, Python variables are containers...")]
        
        # Setup client to return responses in sequence
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Execute with tools
        result = ai_generator.generate_response(
            query="What are Python variables?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python variables",
            course_name="Python Basics"
        )
        
        # Verify final response
        assert "Based on the search results" in result
        
        # Verify client was called twice (initial + final)
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_no_tool_use_response(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test response when no tool use is needed"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock direct response without tool use
        direct_response = Mock()
        direct_response.content = [Mock(text="This is a direct answer without using tools")]
        direct_response.stop_reason = "stop"
        
        mock_client.messages.create.return_value = direct_response
        
        # Execute
        result = ai_generator.generate_response(
            query="What is 2+2?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify no tool execution
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Verify direct response
        assert "This is a direct answer without using tools" in result
        
        # Verify client called only once
        assert mock_client.messages.create.call_count == 1
    
    @patch('anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling of multiple tool calls in one response"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock multiple tool use contents
        tool_content_1 = Mock()
        tool_content_1.type = "tool_use"
        tool_content_1.name = "search_course_content"
        tool_content_1.id = "tool_123"
        tool_content_1.input = {"query": "python basics"}
        
        tool_content_2 = Mock()
        tool_content_2.type = "tool_use"
        tool_content_2.name = "search_course_content"
        tool_content_2.id = "tool_456"
        tool_content_2.input = {"query": "advanced python"}
        
        initial_response = Mock()
        initial_response.content = [tool_content_1, tool_content_2]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Combined results from multiple searches")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        # Execute
        result = ai_generator.generate_response(
            query="Tell me about Python",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="python basics")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="advanced python")
        
        assert "Combined results from multiple searches" in result
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling of tool execution errors"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test query"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="I apologize, there was an error with the search")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Make tool execution return error
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        # Execute
        result = ai_generator.generate_response(
            query="Search for something",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was called despite error
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify response handles error
        assert "I apologize, there was an error" in result
    
    @patch('anthropic.Anthropic')
    def test_malformed_tool_parameters(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling of malformed tool parameters from Claude"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock tool use with malformed parameters
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {
            "query": "test",
            "invalid_param": "should_not_exist"
        }
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Response after handling malformed params")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Tool handled malformed params gracefully"
        
        # Execute - should not raise exception
        result = ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was called with all provided parameters (including invalid ones)
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test",
            invalid_param="should_not_exist"
        )
        
        assert "Response after handling malformed params" in result
    
    @patch('anthropic.Anthropic')
    def test_missing_required_parameters(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling when Claude omits required parameters"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock tool use missing required 'query' parameter
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {
            "course_name": "Python Basics"  # Missing required 'query'
        }
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Handled missing parameters")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Tool handled missing params"
        
        # Execute
        result = ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was called even with missing params
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            course_name="Python Basics"
        )
        
        assert "Handled missing parameters" in result
    
    @patch('anthropic.Anthropic')
    def test_conversation_history_integration(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test tool calling with conversation history"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock direct response (no tool use)
        response = Mock()
        response.content = [Mock(text="Response considering history")]
        response.stop_reason = "stop"
        
        mock_client.messages.create.return_value = response
        
        # Execute with history
        result = ai_generator.generate_response(
            query="Follow up question",
            conversation_history="User: What is Python?\nAssistant: Python is a programming language.",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args['system']
        assert "What is Python?" in call_args['system']
        
        assert "Response considering history" in result
    
    @patch('anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test handling of Anthropic API errors"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock API error
        mock_client.messages.create.side_effect = anthropic.APIError("API rate limit exceeded")
        
        # Execute - should propagate the exception for higher-level handling
        with pytest.raises(anthropic.APIError):
            ai_generator.generate_response(
                query="Test query",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager
            )
    
    @patch('anthropic.Anthropic')
    def test_tool_result_message_format(self, mock_anthropic_class, ai_generator, mock_tool_manager):
        """Test that tool results are properly formatted as messages"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        # Execute
        ai_generator.generate_response(
            query="Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify the second call (final response) has proper message structure
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args['messages']
        
        # Should have: user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'
        assert messages[2]['role'] == 'user'
        
        # Check tool result format
        tool_results = messages[2]['content']
        assert len(tool_results) == 1
        assert tool_results[0]['type'] == 'tool_result'
        assert tool_results[0]['tool_use_id'] == 'tool_123'
        assert tool_results[0]['content'] == 'Tool result content'


class TestAIGeneratorConfiguration:
    """Test AIGenerator configuration and setup"""
    
    def test_initialization_with_valid_config(self):
        """Test proper initialization with valid configuration"""
        generator = AIGenerator(api_key="test-key", model="claude-3-haiku-20240307")
        
        assert generator.model == "claude-3-haiku-20240307"
        assert generator.base_params['model'] == "claude-3-haiku-20240307"
        assert generator.base_params['temperature'] == 0
        assert generator.base_params['max_tokens'] == 800
    
    @patch('anthropic.Anthropic')
    def test_system_prompt_content(self, mock_anthropic_class):
        """Test that system prompt contains expected content"""
        generator = AIGenerator(api_key="test-key", model="claude-3-haiku-20240307")
        
        # Check system prompt contains tool usage guidelines
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "Course Content Search Tool" in generator.SYSTEM_PROMPT
        assert "Course Outline Tool" in generator.SYSTEM_PROMPT
        assert "One tool call per query maximum" in generator.SYSTEM_PROMPT
    
    def test_base_params_structure(self):
        """Test that base parameters are properly structured"""
        generator = AIGenerator(api_key="test-key", model="test-model")
        
        expected_keys = {'model', 'temperature', 'max_tokens'}
        assert set(generator.base_params.keys()) == expected_keys
        
        assert isinstance(generator.base_params['temperature'], (int, float))
        assert isinstance(generator.base_params['max_tokens'], int)
        assert generator.base_params['max_tokens'] > 0