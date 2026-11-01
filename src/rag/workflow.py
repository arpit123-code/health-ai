import logging
from typing import Literal
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from config.settings import get_settings
from pipeline.state import AgentState, RouteDecision, EvidenceGrade
from data.vector_store.start import get_retriever
import os

logger = logging.getLogger(__name__)
settings = get_settings()


_llm = None
_web_search = None

def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=3,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for a health-ai assistant.

Choose exactly one route:

- "kb": The user asks about a health issue, symptom, condition, disease, medication,
  treatment, test result, diet, exercise, sleep, mental health, or wants help
  understanding or diagnosing what they are experiencing. Anything that needs medical
  knowledge to answer should go to kb.

- "direct": Greetings, thanks, goodbyes, casual chat, or questions about what you can do,
  which need no medical knowledge. Also use direct when the answer is already known from
  the conversation so far (for example, a follow-up that just clarifies or repeats
  something you already said).

Rules:
- If the message mixes casual chat with a health question, choose kb.
- If you are unsure between kb and direct, choose kb.
- Do not answer the question yourself. Only return the route.

Question: {state['question']}

Return valid JSON like {{"route":"kb"}}.
""")
    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade = grader.invoke(f"""
You grade evidence for a health-ai assistant.

Question: {state['question']}

Health knowledge base evidence:
{context}

Grade the evidence:

- "good": The evidence directly addresses the health issue in the question and contains
  enough specific information to answer it confidently. This includes relevant symptoms,
  possible causes, recommended care, or guidance that matches what the user described.

- "weak": The evidence is missing, off-topic, only loosely related, too general, or does
  not cover the user's specific symptoms, condition, medication, or situation. Also use
  weak if answering would require guessing beyond what the evidence says.

Rules:
- Judge only on the evidence provided, not on your own medical knowledge.
- If the evidence covers only part of the question, return weak.
- If you are unsure between good and weak, return weak.

Return valid JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")}


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
You grade web search evidence for a health-ai assistant.

Question: {state['question']}

Web evidence:
{state['web_results']}

Grade the evidence:

- "good": The evidence directly addresses the health issue in the question and contains
  enough specific information to answer it confidently. It comes from credible health
  sources (for example, hospitals, government health agencies, medical journals, or
  established medical organizations) and is consistent across results.

- "weak": The evidence is off-topic, only loosely related, too general, or does not cover
  the user's specific symptoms, condition, medication, or situation. Also use weak if it
  comes mainly from low-quality sources (forums, ads, product pages, unsourced blogs),
  if the results contradict each other, or if answering would require guessing beyond
  what the evidence says.

Rules:
- Judge only on the evidence provided, not on your own medical knowledge.
- If the evidence covers only part of the question, return weak.
- If you are unsure between good and weak, return weak.

Return valid JSON like {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this health issue question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful keywords, do not answer, return only the query.
Question: {state['question']}
""").content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(f"""
You are a health information assistant. Answer ONLY from the health knowledge base below.
Be clear, concise, and practical. If steps or care instructions are present, present them
as a short list.

Guidelines:
- Based on the symptoms and details the user gave, explain the most likely possible causes
  that the knowledge base supports. Use wording like "this may be" or "this could be
  consistent with", not definitive statements like "you have".
- Do not invent medical facts, doses, or treatments that are not in the knowledge base.
  If the knowledge base does not cover part of the question, say so plainly.
- If the user's description includes warning signs (for example chest pain, trouble
  breathing, signs of stroke, severe bleeding, fainting, a severe allergic reaction, or
  thoughts of self-harm), tell them to seek immediate medical care or call local
  emergency services (112 in India) before anything else.
- For symptoms that are persistent, worsening, or unclear, recommend seeing a doctor.
- If key details are missing (age, duration, severity, existing conditions, medications),
  ask one or two short follow-up questions.
- Mention that the answer is based on the health knowledge base and is general information,
  not a substitute for professional medical advice.

Question: {state['question']}

Health knowledge base:
{context}
""").content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}


def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are a health information assistant. The health knowledge base did not have enough
information to answer this question. Answer ONLY from the web evidence below.

Start by clearly saying that this answer comes from external web sources, not the health
knowledge base, and that it should be confirmed with a doctor or other qualified
healthcare professional.

Guidelines:
- Based on the symptoms and details the user gave, explain the most likely possible causes
  that the web evidence supports. Use wording like "this may be" or "this could be
  consistent with", not definitive statements like "you have".
- Do not invent medical facts, doses, or treatments that are not in the web evidence.
  If the evidence does not cover part of the question, say so plainly.
- Prefer information from credible health sources (hospitals, government health agencies,
  medical journals, established medical organizations). If the evidence is mixed in
  quality or sources disagree, point that out rather than picking one side.
- If the user's description includes warning signs (for example chest pain, trouble
  breathing, signs of stroke, severe bleeding, fainting, a severe allergic reaction, or
  thoughts of self-harm), tell them to seek immediate medical care or call local
  emergency services (112 in India) before anything else.
- For symptoms that are persistent, worsening, or unclear, recommend seeing a doctor.
- If key details are missing (age, duration, severity, existing conditions, medications),
  ask one or two short follow-up questions.
- Remind the user this is general information, not a substitute for professional medical
  advice.

Question: {state['question']}

Web evidence:
{state['web_results']}
""").content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "Give me the correct problem description and any relevant details, and I will try to help you.",
        "source_used": "insufficient_description",
        "trace": add_trace(state, "Stopped → insufficient description"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)