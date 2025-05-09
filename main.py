from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from IPython.display import display as ipy_display, Image as IPImage
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os
import uuid
from pydantic import BaseModel, Field
from typing import List, TypedDict
from urllib.request import urlopen
import json
import requests
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import pytesseract
from langchain_groq import ChatGroq
import io
from PIL import Image
import numpy as np
import gradio as gr

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Defining the AI agent class
class Billtype(BaseModel):
    typee: int = Field(description="List of BILLS")

class ReimburesemtAgent(TypedDict):
    the_image: bytes
    extracted_text: str
    extracted_price: str
    extracted_bill_type: int
    unique_id: uuid.UUID
    name: str

# Function to create a new CSV file if it does not exist
def creating_new_name(name):
    file_path = f'/content/drive/MyDrive/reimbursement/storage/{name}.csv'
    print(f"Checking if file exists at: {file_path}")  # Debugging line
    if not os.path.exists(file_path):
        print(f"File does not exist. Creating file for: {name}")  # Debugging line
        data = {
            'image': [],
            'extract_text': [],
            'extract_price': [],
            'extract_type': []
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    else:
        print(f"File already exists at: {file_path}")  # Debugging line
    return file_path

# Defining OCR and Agent functions
def run_tesseract(state: ReimburesemtAgent):
    img_byte_arr = state['the_image']
    img = Image.open(io.BytesIO(img_byte_arr))
    text = pytesseract.image_to_string(img)
    new_uuid = uuid.uuid4()
    return {'extracted_text': text, 'unique_id': new_uuid}

def extract_price(state: ReimburesemtAgent):
    text = state['extracted_text']
    prompt = f'''I have extracted text from an OCR model of a bill or receipt of an expense I want to reimburese,
    I want you to tell me the exact total value of the bill.
    return me just a number nothing else
    The extracted text is :{text}
    '''
    response = llm.invoke(prompt)
    return {'extracted_price': response.content}

def extract_type(state: ReimburesemtAgent):
    text = state['extracted_text']
    prompt = f'''I have extracted text from an OCR model of a bill or receipt of an expense I want to reimburese,
    I want you to tell me what type of bill it is.
    return me just the type from the following types:
    1) Fuel
    2) Travel
    3) Hotel
    4) Food
    5) others
    The extracted text is :{text}
    '''
    structure_llm = llm.with_structured_output(Billtype)
    bill: Billtype = structure_llm.invoke(prompt)
    return {'extracted_bill_type': bill.typee}

def saving_in_drive(state: ReimburesemtAgent):
    img = Image.open(io.BytesIO(state['the_image']))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    extract_text = state['extracted_text']
    extract_price = state['extracted_price']
    extract_type = state['extracted_bill_type']
    idd = state['unique_id']
    name = state['name']  # Get the name from user input

    # Log intermediate outputs
    print(f"Extracted Text: {extract_text}")
    print(f"Extracted Price: {extract_price}")
    print(f"Extracted Bill Type: {extract_type}")

    # Map bill type
    if int(extract_type) == 1:
        extract_type = 'Fuel'
    elif int(extract_type) == 2:
        extract_type = 'Travel'
    elif int(extract_type) == 3:
        extract_type = 'Hotel'
    elif int(extract_type) == 4:
        extract_type = 'Food'
    else:
        extract_type = 'other'

    # Check if the file exists, otherwise create a new one
    file_path = creating_new_name(name)  # Ensure the file exists

    # Reading the existing file and appending data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return "Error: No such file found."

    new_row = pd.DataFrame([[img_byte_arr, extract_text, extract_price, extract_type]],
                           columns=['image', 'extract_text', 'extract_price', 'extract_type'])

    # Saving new data to the file
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)  # Update the CSV file

    print(f"Data saved successfully in: {file_path}")

# Defining LangGraph
workflow = StateGraph(ReimburesemtAgent)
workflow.add_node("Running_OCR_for_text", run_tesseract)
workflow.add_node("Extract_price", extract_price)
workflow.add_node("Extract_type", extract_type)
workflow.add_node("save_in_drive", saving_in_drive)

workflow.set_entry_point("Running_OCR_for_text")
workflow.add_edge("Running_OCR_for_text", "Extract_price")
workflow.add_edge("Running_OCR_for_text", "Extract_type")

workflow.add_edge("Extract_price", "save_in_drive")
workflow.add_edge("Extract_type", "save_in_drive")
workflow.add_edge("save_in_drive", END)

memory = MemorySaver()  # This will work after importing MemorySaver

graph_plan = workflow.compile(checkpointer=memory)
ipy_display(IPImage(graph_plan.get_graph(xray=1).draw_mermaid_png()))  # Displaying the graph

# Getting a Gradio instance up
config = {"configurable": {"thread_id": "1"}}

state_input = {
    "the_image": bytearray(),
    "extracted_text": "",
    "extracted_price": "",
    "extracted_bill_type": "",
    "unique_id": "",
    "name": ""  # Empty string to take name from input
}

def process_image(image, name):
    img = Image.open(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    state_input["the_image"] = img_byte_arr
    state_input["name"] = name  # Get the name from the input field

    for event in graph_plan.stream(state_input, config, stream_mode=["updates"]):
        print(f"Current node: {next(iter(event[1]))}")

    try:
        df = pd.read_csv(f'/content/drive/MyDrive/reimbursement/storage/{name}.csv')
        return df[['extract_price', 'extract_type']].to_html()
    except FileNotFoundError:
        return "Error: No such file found."

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath"),  # Changed to filepath for image upload
        gr.Textbox(label="Your Name")  # User input for the name
    ],
    outputs=gr.HTML(),
    title="Image Processor",
    description="Upload an image, it will be processed and the associated data from the excel file will be shown."
)

iface.launch(share=True)  # Now the Gradio interface will provide a public link
