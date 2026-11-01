from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(description="kb for health knowledge base, direct for direct answer generation")

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(description="Whether evidence is sufficient to answer")

class AgentState(TypedDict):
    # Voice & Transcription inputs
    input_audio_path: str
    source_language: str
    raw_transcript: str
    translated_transcript: str
    
    # RAG & Query processing
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    
    # Generation & Voice Output
    answer: str
    translated_answer: str
    output_audio_path: str
    
    # Tracking & Metadata
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[dict]