from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_classic import hub

from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents import AgentExecutor
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input","agent_scratchpad","tool_names"]
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm, 
    tools=tools,
    prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm

def create_reasoning_agent():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
        }
    )
    print(result)

def main():
    print("Hello from agentic-ai-course!")    
    create_reasoning_agent()

if __name__ == "__main__":
    main()


# Simple GEN AI chatbot

# def main():
#     print("Hello from agentic-ai-course!")
#     information = """
#     Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

# Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He received bachelor's degrees from the University of Pennsylvania in 1997 before moving to California, United States, to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

# In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

# Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

# Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
#     """

#     summary_template = """
#     given the information {information} about a person I want you to crate:
#     1. A short summary
#     2. two interesting facts about them
#     """

#     summary_prompt_template = PromptTemplate(
#         input_variables=["information"], template=summary_template
#     )

#     llm = ChatOpenAI(temperature=0, model="gpt-4o")
#     # llm = ChatOllama(temperature=0, model="gemma3:270m")
#     chain = summary_prompt_template | llm

#     response = chain.invoke(input={"information": information})
#     print(response.content)

# Create a web search agent using Tavily

# class Sources(BaseModel):
#     """Schema for a source used by the agent"""
#     url:str = Field(description="The URL of the source")

# class AgentResponse(BaseModel):
#     """Schema for agent response with answer and source"""

#     answer:str = Field(description="The agent's answer to the query")
#     source:List[Sources] = Field(description="List of sources used to genarate answer to the query")

# llm = ChatOpenAI(model="gpt-5")
# tools = [TavilySearch()]
# agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

# def create_web_search_agent():
#     result = agent.invoke(
#         {
#             "messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")
#         }
#     )
#     print(result)