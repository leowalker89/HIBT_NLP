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

from typing import Annotated, Dict, TypedDict, Optional

from langchain_core.messages import BaseMessage

from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
import os
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'

client = pymongo.MongoClient(os.getenv('mdb_uri'))
DB_NAME = "hibt_prod_db"
COLLECTION_NAME = "hibt_prod_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
MONGODB_ANSWER_COLLECTION = client[DB_NAME]["hibt_answer_collection"]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string
        retries: An optional integer representing the number of retries (default: 0)
        document_count: An optional integer representing the document count (default: 0)
    """
    keys: Dict[str, any]
    retries: Optional[int] = 1
    doc_pull_limit: Optional[int] = 4


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The state of the graph
    
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # print("---RETRIEVE---")
    question = state["keys"]["question"]
    limit = state["doc_pull_limit"]
    company = state["keys"]["company"]

    client = pymongo.MongoClient(os.getenv('mdb_uri'))
    DB_NAME = "hibt_prod_db"
    COLLECTION_NAME = "hibt_prod_collection"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    documents = MONGODB_COLLECTION.aggregate([
        {"$vectorSearch": {
            "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
            "path": "embedding",
            "queryVector": embeddings.embed_query(question),
            "numCandidates": 100,
            "limit": limit,
            "filter": {"company": f"{company}"}
        }}
    ])
    state["keys"]["documents"] = [doc['transcript'] for doc in documents]

    return state

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args: 
        state (dict): The state of the graph
    
        Returns:
        state (dict): Updates documents key with relevant documents
    """
    # print("---CHECK RELEVANCE---")
    
    question = state["keys"]["question"]
    documents = state["keys"]["documents"]

    doc_grading_prompt = """ You are evaluating a podcast transcript segment to determine if it contains a discussion relevant to a specific question. Your task is to assess whether the segment includes both a question from the host and an answer from a guest that relate closely to the provided relevant question.
    Transcript segment:
    {transcript}
    Relevant question to compare:
    {question}
    Your job is to decide if the segment discusses a question and answer similar to the relevant question provided. You need to give a binary score indicating the relevance.
    - If the segment contains a question and answer that closely match the relevant question, grade it as 'yes'.
    - If the segment does not contain a question and answer that are relevant, grade it as 'no'.
    Provide your assessment in the following JSON format, with only the 'score' key and your binary decision as the value. No preamble or explanation needed. Make sure to use double quotes instead of single quotes.
    """

    prompt = PromptTemplate(
    template=doc_grading_prompt,
    input_variables=["transcript", "question"]
    )
    llm = ChatAnthropic(model="claude-instant-1.2")
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    chain = prompt | llm | JsonOutputParser()

    relevant_documents = []
    research = "No"
    for doc in documents:
        score = chain.invoke({"question":question, "transcript":doc})
        if score["score"] == "yes":
            relevant_documents.append(doc)
            # print("---GRADE: DOCUMENT RELEVANT---")
        else:
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    if len(relevant_documents) == 0 and state['retries'] >0:
        state["keys"]["research"] = "Yes"
        state['retries'] = state['retries'] - 1
    else:
        state["keys"]["research"] = "No"
    
    state["keys"]["documents"] = relevant_documents
    return state

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    # print("---TRANSFORM QUERY---")
    
    state_dict = state["keys"]
    question = state_dict["question"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n
        Look at the input and try to reason about the underlying sematic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    # LLM
    # llm = ChatMistralAI(
    #     mistral_api_key=os.getenv('MISTRAL_API_KEY'), temperature=0, model="mistral-medium"
    #     )
    llm = ChatAnthropic(model="claude-instant-1.2")

    # print("State dict", state_dict)
    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    state["keys"]["question"] = better_question
    return state

def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    # print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    research = state["keys"]["research"]
    company = state["keys"]["company"]

    if research == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # print("---DECISION: TRANSFORM QUERY and RESEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        # print("---DECISION: GENERATE---")
        return "generate"

from pydantic import BaseModel, Field

class QuestionAnswer(BaseModel):
    # question: str = Field(description="Question that the host asked the guest")
    answer: str = Field(description="The guest's answer to the host's question or 'Question/Answer not found'")


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The state of the graph

    Returns:
        state (dict): New key added to state, generation, that contains generation
    
    """
    # print("---GENERATE---")
    state_dict = state["keys"]
    documents = state_dict["documents"]
    question = state_dict["question"]

    ### everything below this is a placeholder right now, need to come back to this after the grading process

    prompt = """
    You are tasked with analyzing podcast transcripts to extract discussions that are specifically related to a question posed by the host. Here is the question of interest:

    {question}

    Your objective is to find and extract the question that the host asked and the guest's answer that directly responds to this question. 

    Guidelines for the task:
    - Identify the guest's answer that pertains to the question above. The relevant answer is typically located in the same or subsequent paragraphs following the question.
    - Correct common transcription misinterpretations as you extract the guest's response.
    - Exclude the original question from your output. Provide only the guest's answer, ensuring it is word-for-word, with necessary corrections for transcription errors.
    - If a relevant question and answer pair is not found within the provided transcript segment, simply state "Question/Answer not found."

    Please concentrate on extracting the answer from the provided transcript segment below, with attention to detail and accuracy:

    {transcript}

    Respond in the following JSON format.

    {response_template}

    """
    parser = PydanticOutputParser(pydantic_object=QuestionAnswer)

    prompt = PromptTemplate(
    template=prompt,
    input_variables=["transcript", "question"],
    partial_variables={"response_template": parser.get_format_instructions()}
    )
    
    # llm = ChatAnthropic(model="claude-instant-1.2")
    # llm = ChatAnthropic(model="claude-2.0")
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    # llm = ChatMistralAI(
    #     mistral_api_key=os.getenv('MISTRAL_API_KEY'), temperature=0, model="mistral-medium"
    #     )
    # llm = ChatOpenAI(api_key=os.getenv("PPLX_API_KEY"), base_url="https://api.perplexity.ai", model='mixtral-8x7b-instruct')
    # llm = ChatAnthropic(model="claude-instant-1.2")
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | parser

    # Run
    generation = rag_chain.invoke({"transcript": documents, "question": question})

    if generation == "Question/Answer not found.":
        # llm = ChatAnthropic(model="claude-instant-1.2")
        llm = ChatOpenAI(model="gpt-4-0125-preview")
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"transcript": documents, "question": question})
    
    # print(generation)
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }
