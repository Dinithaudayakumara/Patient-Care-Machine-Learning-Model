import os
import traceback
from datetime import datetime
from typing import List

import motor.motor_asyncio
import pandas as pd
from bson import ObjectId
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# show all columns
pd.set_option('display.max_columns', None)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load environment variables
from dotenv import load_dotenv

load_dotenv()

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.test

data_length = 100000

base_url = "http://localhost:8000/"
save_path = "data/"

from sentiment import predict_sentiment


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# feedbackSchema
#     patientID: { type: Schema.Types.ObjectId, ref: "patient" },
#     feedbackDescription: { type: String },
#     feedbackDateandTime: { type: string },
#     sentiment: { type: String },

class FeedbackModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    patientID: PyObjectId = Field(default_factory=PyObjectId)
    feedbackDescription: str
    feedbackDateandTime: datetime = Field(default_factory=datetime.now)
    sentiment: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "patientID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "feedbackDescription": "I am very happy with the service",
                "feedbackDateandTime": "2021-07-19T15:00:00.000Z",
                "sentiment": "positive"
            }
        }


@app.get("/feedbacks", response_description="List all feedbacks", response_model=List[FeedbackModel])
async def list_feedbacks():
    feedbacks = await db["feedbacks"].find().to_list(data_length)
    return feedbacks


@app.get("/feedbacks/{id}", response_description="Get a single feedback", response_model=FeedbackModel)
async def get_feedback(id: str):
    # get feedbackdetils by id
    feedback = await db["feedbacks"].find_one({"_id": ObjectId(id)})
    if feedback:
        return feedback

    return {"error": "feedback not found"}


@app.post("/update-feedback/{order_id}")
async def update_feedback(feedback_id: str):
    try:
        feedback = await get_feedback(feedback_id)
        sentiment = predict_sentiment(feedback["feedbackDescription"])
        await db["feedbacks"].update_one({"_id": ObjectId(feedback["_id"])}, {"$set": {"sentiment": sentiment}})

        return {"sentiment": sentiment}

    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "failed"}
