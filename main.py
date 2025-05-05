from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tools
from crewai import Agent, Crew, Task
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000
)

pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"

if not os.path.exists("attention_is_all_you_need.pdf"):
    response = requests.get(pdf_url)
    with open("attention_is_all_you_need.pdf", "wb") as file:
        file.write(response.content)

rag_tool = PDFSearchTool(
    pdf='attention_is_all_you_need.pdf',
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama3-8b-8192",
                temperature=0.1,
                top_p=1,
                # stream=True
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                # task_type="retrieval document",
                # title="Embeddings"
            )
        )
    )
)

rag_tool.run("How did self attention mechanism evolve in large language models?")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

web_search_tool = TavilySearchResults(k=3)
web_search_tool.run("What is self attention mechanism in large language models?")

def router_tool(question):
    """Router Function"""
    if 'self-attention' in question:
        return 'vectorstore'
    else:
        return 'web_search'
    
Router_agent = Agent(
    role ='router',
    goal = 'Route user question to a vectorstore or web search',
    backstory = (
        "You are an expert at routing a user question to a vectorstore or a web search",
        "Use the vectorestore for questions on concept related to Retrieval Augmented Generation",
        "You do not need to be stringent with the keywords in the question related to the these topic. Otherwise use websearch"
    ),
    verbose = True,
    allow_delegation = False,
    llm = llm
)

Retriever_Agent = Agent(
    role = "Retriever",
    goal = 'Use the information received from the vectorstore to answer the question',
    backstory = (
        "You are an expert for question-answering tasks",
        "Use the information present in the retrieved context to answer the question",
        "You have to provide a clear concise answer"
    ),
    verbose = True,
    allow_delegation = False,
    llm = llm
)

Grader_Agent= Agent(
    role = "Answer Grader",
    goal = 'Filter out enroneous retrievals',
    backstory = (
        "You are a grader assessing relevance of a retrieved document to a user question",
        "If the document contain keywords related to the user question, grade it as relevant",
        "It does not need to be stringent, you will have to make sure the answer is relevant to the question"
    ),
    verbose = True,
    allow_delegation = False,
    llm = llm
)

hallucination_grader = Agent(
    role = "Hallucination Grader",
    goal = 'Filter out hallucination',
    backstory = (
        "You are an hallucination grader assessing whether an answer is grounded in/supported by a set of facts",
        "Make sure you thoroughly review the answer and check if the answer provided is in alignment with the question asked"
    ),
    verbose = True,
    allow_delegation = False,
    llm = llm
)

answer_grader = Agent(
    role = "Answer Grader",
    goal = 'Filter out hallucation from the answer',
    backstory = (
        "You are a grader assessing whether an answer is useful to resolve the user question",
        "Make sure you thoroughly review the answer and check if it makes sense for the question asked",
        "If the answer generated is not relevant then perform a websearch using web_search_tool",
        "If the answer generated is relevant give a clear and concise response"
    ),
    verbose = True,
    allow_delegation = False,
    llm = llm
)

router_task = Task(
    description = (
        "Analyze the keywords in the question: '{question}'. "
        "Based on the keywords decide whether it is eligible for vectorsearch or websearch. "
        "Return a single word 'vectorsearch' if it is eligible for vector search. "
        "Return a single word 'websearch' if it is eligible for web search. "
        "Do not provide any other preamble or explanation."
    ),
    expected_output = (
        "Give a binary choice 'websearch' or 'vectorsearch' based on the question. "
        "Do not provide any other preamble or explanation."
    ),
    agent=Router_agent,
    tools=[router_tool]
)

retriever_task = Task(
    description=(
        "Using the tool provided, retrieve relevant information from the vectorstore "
        "to answer the user question: '{question}'. "
        "Use the retrieval tool to ensure that the answer is factually supported by the document."
    ),
    expected_output=(
        "A concise and clear answer to the question using only the retrieved information from the vectorstore."
    ),
    agent=Retriever_Agent,
    context = [router_task],
    tools=[rag_tool]
)

grader_task = Task(
    description=(
        "Evaluate whether the retrieved content is relevant to the user’s question: '{question}'. "
        "If the document contains semantically matching content, mark it as relevant. "
        "Otherwise, mark it as irrelevant."
    ),
    expected_output=(
        "A verdict: 'Relevant' or 'Irrelevant' along with a brief justification."
    ),
    agent=Grader_Agent,
    context = [retriever_task]
)

hallucination_task = Task(
    description=(
        "Analyze the answer generated for the question: '{question}'. "
        "Compare it with the context retrieved from the document. "
        "Determine if any part of the answer is a hallucination or unsupported by the context."
    ),
    expected_output=(
        "Respond with 'Grounded' if the answer is well supported by the document. "
        "Respond with 'Hallucinated' if the answer includes unsupported claims. Include a short explanation."
    ),
    agent=hallucination_grader,
    context=[grader_task]
)

answer_task = Task(
    description=(
        "Based on the outputs of the hallucination and relevance checks, generate a final answer for the user’s question: '{question}'. "
        "If the answer is found to be relevant and grounded, return it as-is. "
        "If it is not, perform a web search using the tool provided and synthesize a correct answer."
    ),
    expected_output=(
        "A clear and useful answer that directly addresses the user’s question."
    ),
    agent=answer_grader,
    context=[hallucination_task],
    tools=[web_search_tool]
)

crew = Crew(
    agents=[Router_agent, Retriever_Agent, Grader_Agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True
)

question = "How did the self-attention mechanism evolve in large language models?"
crew.kickoff(inputs={"question": question})
