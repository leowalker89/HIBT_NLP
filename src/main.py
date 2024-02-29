import pymongo
from dotenv import load_dotenv
import os

from utils.preprocess_transcription import remove_ads, identify_host, insert_marker_before_host
from utils.extract_comp_guest import re_extract_comp_guest

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI
from langchain_mistralai.chat_models import ChatMistralAI

from typing import Annotated, Dict, TypedDict

from langchain_core.messages import BaseMessage

from langchain.prompts import PromptTemplate
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
import os
import pprint

from utils.company_answer_extraction import retrieve, grade_documents, generate, transform_query, decide_to_generate

from langgraph.graph import END, StateGraph
load_dotenv()

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string
    """
    keys: Dict[str, any]
    retries: int
    document_count: int
    company: str

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
# workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    inputs = {
        "keys": {
            "company": "Audible",
            "question": "In reflecting on your path to success, how do you view the contributions of diligence, intellect, and fortuitous events? Are there particular moments or decisions that highlight the importance of these elements?"
        }
    }

    answer = app.invoke(inputs)
