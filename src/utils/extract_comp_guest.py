from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import re

load_dotenv()

comp_guest_prompt = """You are an expert at parsing things out of a file string. 
    You are given a file string that should contain a company name (could be multiple) and a guest name (could be multiple).
    You are to parse these out of the string name.
    ---
    {response_template}
    ---
    Examples:
    filename: "A biometric smart gun with Kai Kloepfer of Biofire-transcript.txt"
    response: ""company": ["Biofire"], "guest": ["Kai Kloepfer"]"

    filename: "ActOne Group_ Janice Bryant Howroyd (2018)-transcript.txt"
    response: ""company": ["ActOne Group"], "guest": ["Janice Bryant Howroyd"]"

    filename: "HIBT/podscribe_transcription/hibt_main/McBride Sisters Wine (Part 1 of 2)_ Robin McBride and Andréa McBride John-transcript.txt"
    response: ""company": ["McBride Sisters Wine"], "guest": ["Robin McBride", "Andréa McBride John"]"

    filename: "HIBT/podscribe_transcription/hibt_main/reCAPTCHA and Duolingo_ Luis von Ahn-transcript.txt"
    response: ""company": ["reCAPTCHA", "Duolingo"], "guest": ["Luis von Ahn"]"
    ---
    Parse the company and guest names from the following file string:
    {file_string}
"""

class comp_guest(BaseModel):
    company: List[str]
    guest: List[str]


def llm_extract_comp_guest(file_string: str) -> comp_guest:
    # Extracts Company and Guest from the file name using an LLM
    parser = PydanticOutputParser(pydantic_object=comp_guest)
    parser.get_format_instructions()

    prompt_template = PromptTemplate(
            template=comp_guest_prompt,
            input_variables=["file_string"],
            partial_variables={"response_template": parser.get_format_instructions()},
        )
    formatted_input = prompt_template.format_prompt(file_string=file_string)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

    response = llm.invoke(formatted_input.to_string())
    parsed_output = json.loads(parser.parse(response.content).json())

    return parsed_output

def re_extract_comp_guest(file_string: str) -> comp_guest:
    # Extracts Company and Guest from the file name using regex
    if ' with ' in file_string and ' of ' in file_string:
        match = re.match(r'.* with (.*?) of (.*?)-transcript\.txt', file_string)
        guest = match.group(1)
        company = match.group(2)
    else:
        match = re.match(r'(.*?)_ (.*?)-transcript\.txt', file_string)
        company = match.group(1)
        guest = match.group(2)

    result = {
        "company": company,
        "guest": guest
    }
    return result