import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0.5, model_name="llama3-8b-8192")

dream_prompt = PromptTemplate(
    input_variables=["dreams"],
    template="""
    Imagine yourself in the future, having achieved everything you once dreamed of. This is a paragraph written by you, The future self to your past self, reflecting on how you achieved all of your dreams. You will describe a day in your life as the future self, having accomplished all the goals and dreams you once had, and then explain how you reached that point and inspire your past self.

    Here are the dreams you once had:
    {dreams}

    Write the paragraph in simple, plain English, avoiding any fancy words. Don't mention past self or future self in the paragraph, just use "you" and "me" instead. Describe various parts of your day in the future like how the mornings, afternoon and evenings are, what all you have right now etc. The goal is to inspire your past self and reassure them that they do achieve everything they dreamed of.
    At the end of the paragraph always include an inpiration to your past self, telling them that they are going to make it where you are one day, dont say this exact line but improvise
    Remember you are talking to your past self nobody else
    Your response should start with the paragraph itself, don't start with here is a paragraph or anything like that.
    """
)

dream_chain = dream_prompt | llm

random_dream_prompt = PromptTemplate(
    input_variables=[],
    template="""
    Generate a new dream that could inspire someone. Make it a goal or aspiration like "Travel the world and explore cultures" or "Write a bestselling novel one day". The dream should be short, simple, and achievable.
    Your response should start with the dream itself, dont say heres a random dream or anything like that. No text before or after the dream.
    """
)

random_dream_chain = random_dream_prompt | llm

validate_dream_prompt = PromptTemplate(
    input_variables=["dream"],
    template="""
    Please evaluate the following text to determine if it is a valid dream, goal, or aspiration. A valid dream or goal should meet the following criteria:
    
    1. It is expressed as a personal dream, goal, or aspiration.
    2. It is specific and framed in a way that someone could realistically have it as a dream or goal.
    3. It is safe (not harmful, violent, offensive, or inappropriate in any way).

    The text is invalid if:
    1. It is a question, statement, conversation, or anything other than a personal dream or goal.
    2. It includes harmful, violent, offensive, or inappropriate content.

    Dream: {dream}

    If the text meets all the criteria for a valid dream, respond with "valid". 
    If it does not meet the criteria (e.g., it's a question, conversation, or unsafe), respond with "invalid".

    Respond only with "valid" or "invalid" — no other text.
    """
)

validate_dream_chain = validate_dream_prompt | llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://futureself.vercel.app", "https://futureself.vercel.app/chat"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DreamRequest(BaseModel):
    dreams: str

class DreamResponse(BaseModel):
    content: str

@app.post("/dreams", response_model=DreamResponse)
async def generate_dream(dream_request: DreamRequest):
    try:
        res = dream_chain.invoke({"dreams": dream_request.dreams})
        return DreamResponse(content=res.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/random_dream", response_model=DreamResponse)
async def generate_random_dream():
    try:
        res = random_dream_chain.invoke({})
        return DreamResponse(content=res.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_dream", response_model=DreamResponse)
async def validate_and_generate_dream(dream_request: DreamRequest):
    try:
        validation_res = validate_dream_chain.invoke({"dream": dream_request.dreams})
        return DreamResponse(content=validation_res.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#uvicorn app:app --port 8001 --reload
