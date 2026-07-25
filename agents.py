from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx

load_dotenv()


@retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(httpx.HTTPStatusError)
)
def _invoke_with_retry(runnable, *args, **kwargs):
    return runnable.invoke(*args, **kwargs)


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=2
)


def build_search_agent():
    agent = create_agent(
        model=llm,
        tools=[web_search]
    )

    class RetryAgent:
        def __init__(self, a):
            self.a = a

        def invoke(self, *args, **kwargs):
            return _invoke_with_retry(self.a, *args, **kwargs)

    return RetryAgent(agent)


def build_reader_agent():
    agent = create_agent(
        model=llm,
        tools=[scrape_url]
    )

    class RetryAgent:
        def __init__(self, a):
            self.a = a

        def invoke(self, *args, **kwargs):
            return _invoke_with_retry(self.a, *args, **kwargs)

    return RetryAgent(agent)


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])


class RetryChain:
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, *args, **kwargs):
        return _invoke_with_retry(self.chain, *args, **kwargs)


writer_chain = RetryChain(writer_prompt | llm | StrOutputParser())


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = RetryChain(critic_prompt | llm | StrOutputParser())
