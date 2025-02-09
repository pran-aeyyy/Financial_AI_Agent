import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources of the information you find on the web"],
    show_tool_calls=True,
    markdown=True,
)

# Create Financial Agent
financial_agent = Agent(
    name="Financial Agent",
    role="Access financial data and information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the information wherever possible"],
    show_tool_calls=True,
    markdown=True,
)

# Create Multi-Agent
multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Always include the sources of the information you find on the web", "Use tables to display the information wherever possible"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.title("Financial & Web Search AI Agent 🔍")
st.write("Ask any question related to finance or general web search! 🚀")

# User Input
query = st.text_input("Enter your query:", "What is the latest news about NVDA?")
if st.button("Get Response"):
    with st.spinner("Fetching response..."):
        response = multi_ai_agent.run(query)  # Run the agent
        
        # Extracting text response
        if response.outputs:
            result_text = response.outputs[0].content  # Assuming outputs contain a list of responses
        else:
            result_text = "No response received."
        
        st.markdown(result_text)  # Display the response
