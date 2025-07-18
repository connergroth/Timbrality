from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolResult:
    success: bool
    data: Dict[str, Any]
    confidence: float = 1.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        self.metadata["timestamp"] = datetime.now().isoformat()


class BaseTool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, intent_result, context) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def _create_error_result(self, error_message: str) -> ToolResult:
        """Create a standardized error result."""
        return ToolResult(
            success=False,
            data={},
            confidence=0.0,
            error=error_message,
            metadata={"tool": self.name}
        )
    
    def _create_success_result(
        self, 
        data: Dict[str, Any], 
        confidence: float = 1.0
    ) -> ToolResult:
        """Create a standardized success result."""
        return ToolResult(
            success=True,
            data=data,
            confidence=confidence,
            metadata={"tool": self.name}
        )