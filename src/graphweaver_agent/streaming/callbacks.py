"""
# /home/abdul/graphweaver-agent/src/graphweaver_agent/streaming/callbacks.py
#
Streaming callback handlers for LangChain/LangGraph agents.
These handlers intercept events and stream output in real-time.
"""
import sys
import json
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish


# =============================================================================
# ANSI Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    @classmethod
    def disable(cls):
        """Disable all colors (for non-terminal output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, "")


# =============================================================================
# Stream Event Types
# =============================================================================

@dataclass
class StreamEvent:
    """Represents a streaming event."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


# =============================================================================
# Base Streaming Handler
# =============================================================================

class StreamingCallbackHandler(BaseCallbackHandler, ABC):
    """
    Abstract base class for streaming callback handlers.
    
    Inherit from this class to create custom streaming behaviors.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._events: List[StreamEvent] = []
    
    @abstractmethod
    def on_stream_token(self, token: str) -> None:
        """Called when a new token is streamed from the LLM."""
        pass
    
    @abstractmethod
    def on_tool_start_stream(self, tool_name: str, inputs: Dict[str, Any]) -> None:
        """Called when a tool starts executing."""
        pass
    
    @abstractmethod
    def on_tool_output_stream(self, output: str, is_final: bool = False) -> None:
        """Called with tool output (can be called multiple times for streaming tools)."""
        pass
    
    @abstractmethod
    def on_tool_end_stream(self, tool_name: str) -> None:
        """Called when a tool finishes executing."""
        pass
    
    # LangChain callback interface implementation
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self._events.append(StreamEvent("token", data={"token": token}))
        self.on_stream_token(token)
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs,
    ) -> None:
        """Handle tool start."""
        tool_name = serialized.get("name", "unknown")
        
        # Parse input
        try:
            inputs = json.loads(input_str) if isinstance(input_str, str) else input_str
        except:
            inputs = {"input": input_str}
        
        self._events.append(StreamEvent("tool_start", data={"tool": tool_name, "inputs": inputs}))
        self.on_tool_start_stream(tool_name, inputs)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Handle tool end."""
        self._events.append(StreamEvent("tool_end", data={"output": output}))
        self.on_tool_output_stream(output, is_final=True)
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Handle agent action."""
        self._events.append(StreamEvent("agent_action", data={
            "tool": action.tool,
            "input": action.tool_input,
            "log": action.log,
        }))
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Handle agent finish."""
        self._events.append(StreamEvent("agent_finish", data={
            "output": finish.return_values,
            "log": finish.log,
        }))
    
    def get_events(self) -> List[StreamEvent]:
        """Get all recorded events."""
        return self._events
    
    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events = []


# =============================================================================
# Console Streaming Handler (prints to terminal)
# =============================================================================

class ConsoleStreamingHandler(StreamingCallbackHandler):
    """
    Streams output directly to the console with colors and formatting.
    
    This is the main handler for interactive terminal sessions.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        show_tool_output: bool = True,
        stream_tool_output: bool = True,
        token_color: str = "",
        tool_name_color: str = Colors.CYAN,
        tool_output_color: str = Colors.GREEN,
        use_colors: bool = True,
    ):
        super().__init__(verbose)
        self.show_tool_output = show_tool_output
        self.stream_tool_output = stream_tool_output
        self.token_color = token_color
        self.tool_name_color = tool_name_color
        self.tool_output_color = tool_output_color
        self.use_colors = use_colors
        
        self._current_tool: Optional[str] = None
        self._tool_output_buffer: str = ""
        
        if not use_colors:
            Colors.disable()
    
    def _print(self, text: str, color: str = "", end: str = "\n"):
        """Print with optional color."""
        if self.use_colors and color:
            print(f"{color}{text}{Colors.RESET}", end=end, flush=True)
        else:
            print(text, end=end, flush=True)
    
    def on_stream_token(self, token: str) -> None:
        """Print token as it streams."""
        self._print(token, self.token_color, end="")
    
    def on_tool_start_stream(self, tool_name: str, inputs: Dict[str, Any]) -> None:
        """Print tool invocation."""
        self._current_tool = tool_name
        self._tool_output_buffer = ""
        
        if self.verbose:
            self._print(f"\n🔧 ", Colors.YELLOW, end="")
            self._print(f"Calling: ", Colors.DIM, end="")
            self._print(f"{tool_name}", self.tool_name_color, end="")
            
            if inputs:
                # Format args nicely
                args_parts = []
                for k, v in inputs.items():
                    v_str = repr(v)
                    if len(v_str) > 50:
                        v_str = v_str[:47] + "..."
                    args_parts.append(f"{k}={v_str}")
                args_str = ", ".join(args_parts)
                self._print(f"({args_str})", Colors.DIM)
            else:
                print()
    
    def on_tool_output_stream(self, output: str, is_final: bool = False) -> None:
        """Print tool output (streaming or final)."""
        if not self.show_tool_output:
            return
        
        if self.stream_tool_output and not is_final:
            # Incremental streaming - print new content
            new_content = output[len(self._tool_output_buffer):]
            if new_content:
                # Indent output and handle newlines
                for char in new_content:
                    if char == "\n":
                        print()
                        self._print("   ", "", end="")
                    else:
                        self._print(char, self.tool_output_color, end="")
                self._tool_output_buffer = output
        elif is_final:
            # Final output - print everything with proper formatting
            self._print("   ", "", end="")
            lines = output.split("\n")
            for i, line in enumerate(lines):
                self._print(line, self.tool_output_color, end="")
                if i < len(lines) - 1:
                    print()
                    self._print("   ", "", end="")
            print()
    
    def on_tool_end_stream(self, tool_name: str) -> None:
        """Handle tool completion."""
        self._current_tool = None
        self._tool_output_buffer = ""


# =============================================================================
# Tool Streaming Handler (for tools that produce incremental output)
# =============================================================================

class ToolStreamingHandler:
    """
    Helper class to stream output from within a tool.
    
    Usage in a tool:
    
    @tool
    def my_tool(query: str) -> str:
        streamer = ToolStreamingHandler()
        
        streamer.start("Processing query...")
        for chunk in process_chunks(query):
            streamer.stream(chunk)
        streamer.end()
        
        return streamer.get_output()
    """
    
    def __init__(
        self,
        console_output: bool = True,
        indent: str = "   ",
        color: str = Colors.GREEN,
    ):
        self.console_output = console_output
        self.indent = indent
        self.color = color
        
        self._output_buffer: List[str] = []
        self._started = False
    
    def start(self, message: str = None) -> None:
        """Signal start of tool execution."""
        self._started = True
        if message and self.console_output:
            print(f"{self.indent}{Colors.DIM}{message}{Colors.RESET}", flush=True)
    
    def stream(self, content: str) -> None:
        """Stream a chunk of output."""
        if not self._started:
            self.start()
        
        self._output_buffer.append(content)
        
        if self.console_output:
            # Handle newlines with proper indentation
            for char in content:
                if char == "\n":
                    print()
                    print(self.indent, end="", flush=True)
                else:
                    print(f"{self.color}{char}{Colors.RESET}", end="", flush=True)
    
    def stream_line(self, line: str) -> None:
        """Stream a complete line of output."""
        self.stream(line + "\n")
    
    def end(self, message: str = None) -> None:
        """Signal end of tool execution."""
        if message and self.console_output:
            print(f"\n{self.indent}{Colors.DIM}{message}{Colors.RESET}", flush=True)
        elif self.console_output:
            print(flush=True)
        self._started = False
    
    def get_output(self) -> str:
        """Get the complete buffered output."""
        return "".join(self._output_buffer)
    
    def clear(self) -> None:
        """Clear the output buffer."""
        self._output_buffer = []
        self._started = False


# =============================================================================
# WebSocket Streaming Handler (for web applications)
# =============================================================================

class WebSocketStreamingHandler(StreamingCallbackHandler):
    """
    Streams output via WebSocket for web applications.
    
    This handler sends JSON events that can be consumed by a frontend.
    """
    
    def __init__(
        self,
        send_func: Callable[[str], None],
        verbose: bool = True,
    ):
        """
        Args:
            send_func: Function to send data (e.g., websocket.send)
            verbose: Whether to include verbose events
        """
        super().__init__(verbose)
        self.send = send_func
    
    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event via the send function."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        self.send(json.dumps(event))
    
    def on_stream_token(self, token: str) -> None:
        """Emit token event."""
        self._emit("token", {"token": token})
    
    def on_tool_start_stream(self, tool_name: str, inputs: Dict[str, Any]) -> None:
        """Emit tool start event."""
        self._emit("tool_start", {
            "tool": tool_name,
            "inputs": inputs,
        })
    
    def on_tool_output_stream(self, output: str, is_final: bool = False) -> None:
        """Emit tool output event."""
        self._emit("tool_output", {
            "output": output,
            "is_final": is_final,
        })
    
    def on_tool_end_stream(self, tool_name: str) -> None:
        """Emit tool end event."""
        self._emit("tool_end", {"tool": tool_name})


# =============================================================================
# Callback Handler Factory
# =============================================================================

def create_streaming_handler(
    mode: str = "console",
    **kwargs,
) -> StreamingCallbackHandler:
    """
    Factory function to create streaming handlers.
    
    Args:
        mode: One of "console", "websocket", "json"
        **kwargs: Additional arguments for the handler
    
    Returns:
        A configured streaming handler
    """
    handlers = {
        "console": ConsoleStreamingHandler,
        "websocket": WebSocketStreamingHandler,
    }
    
    handler_class = handlers.get(mode)
    if handler_class is None:
        raise ValueError(f"Unknown handler mode: {mode}. Available: {list(handlers.keys())}")
    
    return handler_class(**kwargs)