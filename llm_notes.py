import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import google.genai as genai

# load .env file
load_dotenv()

# access .env variables
api_key = os.getenv("API_KEY")

# client = genai.Client(apikey=) to start an authorized session
# client.models.generate_content(model='model', contents='prompt', config=config) => print(response)
# client.models.generate_content(config=genai.types.GenerateContentConfig(system_instruction="instructions for gemini to follow when answering prompts."))
# conversation history is not remembered by Gemini, so you need to store it and pass it with the contens prompt