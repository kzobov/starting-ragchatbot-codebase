import anthropic
from typing import List, Optional, Dict, Any
import time
from dataclasses import dataclass, field


@dataclass
class RoundState:
    """Tracks execution state and progress across sequential tool calling rounds"""
    round_number: int = 0
    total_tools_executed: int = 0
    tools_per_round: List[List[Dict]] = field(default_factory=list)
    errors_encountered: List[Dict] = field(default_factory=list)
    sources_accumulated: List[Dict] = field(default_factory=list)
    
    def advance_round(self):
        """Move to the next round"""
        self.round_number += 1
        self.tools_per_round.append([])
        
    def add_tool_execution(self, tool_name: str, result: str, success: bool = True):
        """Track a tool execution in the current round"""
        if not self.tools_per_round:
            self.tools_per_round.append([])
            
        self.tools_per_round[-1].append({
            'tool': tool_name,
            'result_length': len(result),
            'success': success,
            'timestamp': time.time()
        })
        self.total_tools_executed += 1
        
    def add_error(self, error: str, round_num: int):
        """Record an error for a specific round"""
        self.errors_encountered.append({
            'round': round_num,
            'error': error,
            'timestamp': time.time()
        })
        
    def add_sources(self, sources: List[Dict], round_num: int):
        """Add sources from a round with deduplication"""
        dedup_keys = {f"{s.get('course_title', '')}_{s.get('lesson_number', '')}" 
                     for s in self.sources_accumulated}
        
        for source in sources:
            dedup_key = f"{source.get('course_title', '')}_{source.get('lesson_number', '')}"
            if dedup_key not in dedup_keys:
                source_with_round = source.copy()
                source_with_round['discovered_in_round'] = round_num
                self.sources_accumulated.append(source_with_round)
                dedup_keys.add(dedup_key)


class SequentialMessageChain:
    """Manages conversation state across multiple tool calling rounds"""
    
    def __init__(self, system_prompt: str, max_rounds: int = 2):
        self.messages = []
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.round_state = RoundState()
        
    def add_user_message(self, content: str):
        """Add a user message to the conversation"""
        self.messages.append({"role": "user", "content": content})
        
    def add_assistant_message(self, content: List):
        """Add an assistant message (including tool use) to the conversation"""
        self.messages.append({"role": "assistant", "content": content})
        
    def add_tool_results(self, tool_results: List[Dict]):
        """Add tool execution results to the conversation"""
        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})
            
    def get_messages(self) -> List[Dict]:
        """Get all messages in the conversation chain"""
        return self.messages.copy()
        
    def should_continue_rounds(self) -> bool:
        """Check if more rounds are allowed"""
        return self.round_state.round_number < self.max_rounds
        
    def advance_round(self):
        """Move to the next round"""
        self.round_state.advance_round()
        
    def get_accumulated_sources(self) -> List[Dict]:
        """Get all sources accumulated across rounds"""
        return self.round_state.sources_accumulated


class SequentialToolError(Exception):
    """Exception for sequential tool calling errors"""
    def __init__(self, message: str, round_num: int = 0, tool_name: str = ""):
        self.message = message
        self.round_num = round_num  
        self.tool_name = tool_name
        super().__init__(self.message)


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

SEQUENTIAL TOOL CALLING PROTOCOL:
- You have up to 2 rounds to use tools for complex queries
- Round 1: Initial tool calls based on user query
- Round 2: Additional tool calls based on Round 1 results (if needed)
- Synthesize information from ALL rounds into your final response

Tool Usage Guidelines:
- **Course Content Search Tool**: Use for questions about specific course content, detailed materials, or when you need to search within course materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, course overviews, or when users ask "what lessons are in X course" or "show me the outline of Y course"
- **Sequential Strategy**: Use initial results to inform follow-up searches for complex queries
- If tools yield no results, state this clearly without offering alternatives

MULTI-STEP REASONING GUIDELINES:
- If initial search yields partial results, consider broader/narrower searches in next round
- If course outline is requested, use outline tool first, then search for specific content if needed  
- If comparing courses, search each course separately then synthesize findings
- If finding relationships between lessons, search multiple lessons then analyze connections
- Use course outline results to guide subsequent targeted searches

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search tool, analyze results, make follow-up searches if beneficial
- **Course structure/outline questions**: Use outline tool first, then present COMPLETE formatted outline exactly as returned
- **Complex queries**: Break down into sequential tool calls as needed
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "rounds" or "tool execution process" to user
 - Present unified response as if from single analysis
 - For outline queries: Present the complete formatted tool output without summarizing

RESPONSE SYNTHESIS:
- Integrate findings from all tool executions across rounds
- Provide complete, coherent answers drawing from all rounds
- Present unified response without mentioning sequential process

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def generate_response_sequential(self, query: str, 
                                   conversation_history: Optional[str] = None,
                                   tools: Optional[List] = None, 
                                   tool_manager=None) -> str:
        """
        Generate AI response with up to 2 sequential tool calling rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build enhanced system content for sequential reasoning
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize message chain for sequential rounds
        message_chain = SequentialMessageChain(system_content, max_rounds=2)
        message_chain.add_user_message(query)
        
        try:
            # Execute sequential rounds (up to 2)
            for round_num in range(1, 3):  # Maximum 2 rounds
                message_chain.advance_round()
                
                response = self._execute_round(message_chain, tools, tool_manager, round_num)
                
                if response.stop_reason != "tool_use":
                    # No more tool calls needed - return response
                    return response.content[0].text
                
                # Execute tools and prepare for next round
                tool_results, round_sources = self._execute_tools_for_round(
                    response, tool_manager, round_num, message_chain.round_state
                )
                
                # Add response and tool results to message chain
                message_chain.add_assistant_message(response.content)
                message_chain.add_tool_results(tool_results)
                
                # Track sources from this round
                if round_sources:
                    message_chain.round_state.add_sources(round_sources, round_num)
                
                # Check if we should continue to next round
                if not message_chain.should_continue_rounds():
                    break
            
            # Final round without tools if max rounds reached
            return self._execute_final_round(message_chain)
            
        except Exception as e:
            error_msg = f"Sequential tool calling failed: {str(e)}"
            print(f"AIGenerator sequential error: {error_msg}")
            
            # Try to provide partial results if available
            if message_chain.round_state.round_number > 0:
                return "I was able to gather some information, though I encountered issues completing my analysis. Please try rephrasing your question."
            else:
                return "I encountered a technical issue while processing your query. Please try again."
    
    def _execute_round(self, message_chain: SequentialMessageChain, tools: List, 
                      tool_manager, round_num: int):
        """Execute a single round of Claude API call"""
        
        api_params = {
            **self.base_params,
            "messages": message_chain.get_messages(),
            "system": message_chain.system_prompt
        }
        
        # Add tools if available and not exceeded max rounds
        if tools and message_chain.should_continue_rounds():
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        try:
            return self.client.messages.create(**api_params)
        except Exception as e:
            raise SequentialToolError(f"Round {round_num} API call failed: {str(e)}", round_num)
    
    def _execute_tools_for_round(self, response, tool_manager, round_num: int, 
                               round_state: RoundState) -> tuple[List[Dict], List[Dict]]:
        """Execute tools for a specific round and return results with sources"""
        
        tool_results = []
        round_sources = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    # Track successful tool execution
                    round_state.add_tool_execution(content_block.name, tool_result, success=True)
                    
                    # Get sources from this tool execution
                    if hasattr(tool_manager, 'get_last_sources'):
                        tool_sources = tool_manager.get_last_sources()
                        if tool_sources:
                            round_sources.extend(tool_sources)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    # Track failed tool execution
                    round_state.add_tool_execution(content_block.name, str(e), success=False)
                    round_state.add_error(str(e), round_num)
                    
                    # Add error result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        return tool_results, round_sources
    
    def _execute_final_round(self, message_chain: SequentialMessageChain) -> str:
        """Execute final round without tools to get synthesized response"""
        
        api_params = {
            **self.base_params,
            "messages": message_chain.get_messages(),
            "system": message_chain.system_prompt
            # No tools for final synthesis round
        }
        
        try:
            final_response = self.client.messages.create(**api_params)
            return final_response.content[0].text
        except Exception as e:
            # Return partial results if available
            if message_chain.round_state.round_number > 0:
                return "I gathered some information but encountered issues providing a complete response."
            else:
                return "I encountered a technical issue. Please try again."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text