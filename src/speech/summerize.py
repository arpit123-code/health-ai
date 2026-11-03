from src.pipeline.state import AgentState
from langchain_groq import ChatGroq
from config.settings import get_settings
from langchain_core.prompts import ChatPromptTemplate
settings=get_settings()

llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.0
)

def summarize_text(state: AgentState) -> str:

    ans=state["answer"]
    summarizer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert healthcare response summarizer for a
        multilingual AI healthcare assistant.

        Your task is to summarize the answer provided in the
        state while preserving ALL medically relevant information.

        IMPORTANT INSTRUCTIONS:

        1. Preserve all important information from the original answer.
        2. Include relevant symptoms, possible causes, explanations,
           precautions, recommendations and next steps.
        3. Never remove emergency warnings or red-flag symptoms.
        4. Preserve medication names, dosages, frequencies and
           restrictions exactly as provided. Never invent or modify them.
        5. Preserve important medical conditions and risk factors.
        6. Include when the user should consult a healthcare professional.
        7. Do not introduce information that is absent from the
           original answer.
        8. Remove repetitive sentences, unnecessary explanations,
           redundant examples and irrelevant details.
        9. Use simple, natural, conversational language suitable
           for text-to-speech.
        10. Organize the response logically so it is easy to understand.

        OUTPUT REQUIREMENTS:

        - Provide a concise but complete response.
        - Do not sacrifice important information just to make
          the response shorter.
        - Use short, natural sentences.
        - Avoid markdown tables, complex formatting and unnecessary headings.
        - Do not add greetings or introductory statements.
        - Return ONLY the summarized answer.
        """
    ),
    (
        "human",
        """
        Original healthcare answer:

        {answer}

        Create a concise summary that retains every relevant
        medical detail and important recommendation.
        """
    )
])
    chain = summarizer_prompt | llm
    response = chain.invoke({"answer": ans})
    return response.content.strip()


    