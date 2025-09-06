"""
Integration tests for AIGenerator tool calling functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic
from ai_generator import AIGenerator, SequentialMessageChain, RoundState, SequentialToolError
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


class TestSequentialToolCalling:
    """Test cases for sequential tool calling functionality"""
    
    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance with test API key"""
        return AIGenerator(api_key="test-api-key", model="claude-3-haiku-20240307")
    
    @pytest.fixture
    def mock_tool_manager_with_sources(self):
        """Create mock tool manager that tracks sources"""
        tool_manager = Mock(spec=ToolManager)
        tool_manager.get_tool_definitions.return_value = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"}
                },
                "required": ["query"]
            }
        }]
        
        # Mock different results for each call
        results = ["Round 1 search results", "Round 2 search results"]
        tool_manager.execute_tool.side_effect = results
        
        # Mock sources for each round
        sources_by_call = [
            [{'course_title': 'Python Basics', 'lesson_number': 1}],
            [{'course_title': 'Advanced Python', 'lesson_number': 2}]
        ]
        tool_manager.get_last_sources.side_effect = sources_by_call
        
        return tool_manager
    
    @patch('anthropic.Anthropic')
    def test_successful_two_round_sequential_calling(self, mock_anthropic_class, ai_generator, mock_tool_manager_with_sources):
        """Test successful 2-round sequential tool calling"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Round 1: Tool use response
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.id = "tool_round1"
        round1_tool.input = {"query": "python basics"}
        
        round1_response = Mock()
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"
        
        # Round 2: Another tool use response
        round2_tool = Mock()
        round2_tool.type = "tool_use"  
        round2_tool.name = "search_course_content"
        round2_tool.id = "tool_round2"
        round2_tool.input = {"query": "advanced python"}
        
        round2_response = Mock()
        round2_response.content = [round2_tool]
        round2_response.stop_reason = "tool_use"
        
        # Final response after 2 rounds
        final_response = Mock()
        final_response.content = [Mock(text="Based on both searches, here's the complete answer...")]
        final_response.stop_reason = "stop"
        
        # Setup client to return responses in sequence: Round1, Round2, Final
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Execute sequential tool calling
        result = ai_generator.generate_response_sequential(
            query="Compare Python basics with advanced concepts",
            tools=mock_tool_manager_with_sources.get_tool_definitions(),
            tool_manager=mock_tool_manager_with_sources
        )
        
        # Verify two tool executions occurred
        assert mock_tool_manager_with_sources.execute_tool.call_count == 2
        
        # Verify tool calls had correct parameters
        call1_args = mock_tool_manager_with_sources.execute_tool.call_args_list[0]
        call2_args = mock_tool_manager_with_sources.execute_tool.call_args_list[1]
        
        assert call1_args[0] == ("search_course_content",)
        assert call1_args[1] == {"query": "python basics"}
        
        assert call2_args[0] == ("search_course_content",)
        assert call2_args[1] == {"query": "advanced python"}
        
        # Verify final response includes synthesis
        assert "Based on both searches" in result
        
        # Verify 3 API calls were made (round1, round2, final)
        assert mock_client.messages.create.call_count == 3
    
    @patch('anthropic.Anthropic')
    def test_early_termination_no_tool_use_round2(self, mock_anthropic_class, ai_generator, mock_tool_manager_with_sources):
        """Test early termination when Claude doesn't use tools in round 2"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Round 1: Tool use response
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.id = "tool_round1"
        round1_tool.input = {"query": "test query"}
        
        round1_response = Mock()
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"
        
        # Round 2: Direct answer without tool use
        round2_response = Mock()
        round2_response.content = [Mock(text="Direct answer without using more tools")]
        round2_response.stop_reason = "stop"
        
        # Setup client responses
        mock_client.messages.create.side_effect = [round1_response, round2_response]
        
        # Execute
        result = ai_generator.generate_response_sequential(
            query="Simple question",
            tools=mock_tool_manager_with_sources.get_tool_definitions(),
            tool_manager=mock_tool_manager_with_sources
        )
        
        # Verify only one tool execution
        assert mock_tool_manager_with_sources.execute_tool.call_count == 1
        
        # Verify only 2 API calls (no final synthesis needed)
        assert mock_client.messages.create.call_count == 2
        
        # Verify result
        assert "Direct answer without using more tools" in result
    
    @patch('anthropic.Anthropic')
    def test_first_call_informs_second_call(self, mock_anthropic_class, ai_generator, mock_tool_manager_with_sources):
        """Test that first tool result influences second tool call parameters"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Round 1: Search for course outline
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "search_course_content"
        round1_tool.id = "tool_round1"
        round1_tool.input = {"query": "python course outline"}
        
        round1_response = Mock()
        round1_response.content = [round1_tool]
        round1_response.stop_reason = "tool_use"
        
        # Round 2: Use information from round 1 result
        round2_tool = Mock()
        round2_tool.type = "tool_use"  
        round2_tool.name = "search_course_content"
        round2_tool.id = "tool_round2"
        round2_tool.input = {"query": "lesson 3 details", "course_name": "Python Basics"}
        
        round2_response = Mock()
        round2_response.content = [round2_tool]
        round2_response.stop_reason = "tool_use"
        
        # Final synthesis
        final_response = Mock()
        final_response.content = [Mock(text="Based on the outline and lesson details...")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Execute
        result = ai_generator.generate_response_sequential(
            query="What comes after the basics in the Python course?",
            tools=mock_tool_manager_with_sources.get_tool_definitions(),
            tool_manager=mock_tool_manager_with_sources
        )
        
        # Verify both tools were called with different, informed parameters
        call1_args = mock_tool_manager_with_sources.execute_tool.call_args_list[0][1]
        call2_args = mock_tool_manager_with_sources.execute_tool.call_args_list[1][1]
        
        assert call1_args["query"] == "python course outline"
        assert call2_args["query"] == "lesson 3 details"
        assert call2_args["course_name"] == "Python Basics"
        
        assert "Based on the outline and lesson details" in result
    
    @patch('anthropic.Anthropic')
    def test_max_rounds_termination(self, mock_anthropic_class, ai_generator, mock_tool_manager_with_sources):
        """Test system terminates after exactly 2 rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Create responses that would request tools in every round
        round1_response = Mock()
        round1_response.content = [Mock(type="tool_use", name="search_course_content", id="tool1", input={"query": "test1"})]
        round1_response.stop_reason = "tool_use"
        
        round2_response = Mock()
        round2_response.content = [Mock(type="tool_use", name="search_course_content", id="tool2", input={"query": "test2"})]
        round2_response.stop_reason = "tool_use"
        
        # Final synthesis (no tools allowed)
        final_response = Mock()
        final_response.content = [Mock(text="Final answer after 2 rounds")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Execute
        result = ai_generator.generate_response_sequential(
            query="Complex multi-step query",
            tools=mock_tool_manager_with_sources.get_tool_definitions(),
            tool_manager=mock_tool_manager_with_sources
        )
        
        # Verify exactly 2 tool executions (max rounds)
        assert mock_tool_manager_with_sources.execute_tool.call_count == 2
        
        # Verify 3 API calls: round1, round2, final synthesis
        assert mock_client.messages.create.call_count == 3
        
        # Verify final call has no tools
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        assert 'tools' not in final_call_args
        
        assert "Final answer after 2 rounds" in result
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_failure_in_round2(self, mock_anthropic_class, ai_generator):
        """Test handling of tool execution failure in second round"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Setup tool manager with failure in second call
        tool_manager = Mock()
        tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        tool_manager.execute_tool.side_effect = ["Success in round 1", Exception("Tool failed in round 2")]
        tool_manager.get_last_sources.return_value = []
        
        # Mock responses
        round1_response = Mock()
        round1_response.content = [Mock(type="tool_use", name="search", id="tool1", input={"query": "test1"})]
        round1_response.stop_reason = "tool_use"
        
        round2_response = Mock()
        round2_response.content = [Mock(type="tool_use", name="search", id="tool2", input={"query": "test2"})]
        round2_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock(text="Response with partial results")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Execute - should not raise exception
        result = ai_generator.generate_response_sequential(
            query="Test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify both tools were attempted
        assert tool_manager.execute_tool.call_count == 2
        
        # Verify we got a response despite the failure
        assert isinstance(result, str)
        assert len(result) > 0
    
    @patch('anthropic.Anthropic')
    def test_api_failure_in_round2(self, mock_anthropic_class, ai_generator, mock_tool_manager_with_sources):
        """Test handling of API failure during second round"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client
        
        # Round 1 succeeds, Round 2 fails
        round1_response = Mock()
        round1_response.content = [Mock(type="tool_use", name="search", id="tool1", input={"query": "test"})]
        round1_response.stop_reason = "tool_use"
        
        mock_client.messages.create.side_effect = [
            round1_response,
            Exception("API error in round 2")
        ]
        
        # Execute - should handle error gracefully
        result = ai_generator.generate_response_sequential(
            query="Test query",
            tools=mock_tool_manager_with_sources.get_tool_definitions(),
            tool_manager=mock_tool_manager_with_sources
        )
        
        # Should return partial results message
        assert "issues completing my analysis" in result or "technical issue" in result
    
    def test_message_chain_accumulation(self):
        """Test that SequentialMessageChain properly accumulates context"""
        chain = SequentialMessageChain("Test system prompt")
        
        # Add initial user message
        chain.add_user_message("Initial query")
        assert len(chain.get_messages()) == 1
        assert chain.get_messages()[0]["role"] == "user"
        
        # Add assistant tool use
        tool_content = [{"type": "tool_use", "name": "search", "input": {"query": "test"}}]
        chain.add_assistant_message(tool_content)
        assert len(chain.get_messages()) == 2
        assert chain.get_messages()[1]["role"] == "assistant"
        
        # Add tool results
        tool_results = [{"type": "tool_result", "tool_use_id": "123", "content": "results"}]
        chain.add_tool_results(tool_results)
        assert len(chain.get_messages()) == 3
        assert chain.get_messages()[2]["role"] == "user"
        
    def test_round_state_tracking(self):
        """Test RoundState properly tracks execution progress"""
        state = RoundState()
        
        # Initial state
        assert state.round_number == 0
        assert state.total_tools_executed == 0
        
        # Advance to round 1
        state.advance_round()
        assert state.round_number == 1
        assert len(state.tools_per_round) == 1
        
        # Add tool execution
        state.add_tool_execution("search_tool", "result", success=True)
        assert state.total_tools_executed == 1
        assert len(state.tools_per_round[0]) == 1
        assert state.tools_per_round[0][0]["tool"] == "search_tool"
        
        # Add error
        state.add_error("Test error", 1)
        assert len(state.errors_encountered) == 1
        assert state.errors_encountered[0]["error"] == "Test error"
        
    def test_source_deduplication(self):
        """Test that sources are deduplicated across rounds"""
        state = RoundState()
        
        # Add sources from round 1
        round1_sources = [
            {'course_title': 'Python Basics', 'lesson_number': 1},
            {'course_title': 'Python Advanced', 'lesson_number': 1}
        ]
        state.add_sources(round1_sources, 1)
        assert len(state.sources_accumulated) == 2
        
        # Add overlapping sources from round 2
        round2_sources = [
            {'course_title': 'Python Basics', 'lesson_number': 1},  # Duplicate
            {'course_title': 'Python Advanced', 'lesson_number': 2}  # New
        ]
        state.add_sources(round2_sources, 2)
        
        # Should only have 3 unique sources (duplicate filtered out)
        assert len(state.sources_accumulated) == 3
        
        # Check that round tracking is added
        assert all('discovered_in_round' in source for source in state.sources_accumulated)