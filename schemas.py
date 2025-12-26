from typing import List
from pydantic import BaseModel, Field

class Sources(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and source"""

    answer:str = Field(description="The agent's answer to the query")
    source:List[Sources] = Field(description="List of sources used to genarate answer to the query")