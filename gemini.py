import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI


def serialize_object(obj):
    """Custom serialization for objects that are not natively serializable by JSON."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def generate_context(context, given_alert):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBET2W7lgxi2_600MBj9UQJ5KgQ5mUrcgk"

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    request_data = {
        "Given_alert": given_alert,
        "similiar_alerts": context,
        "task": "You are an AI security analyst. Analyze the given alert information and information about similar alerts. Give possible alert risk and triage steps in crisp steps",
    }

    try:
        request_json = json.dumps(request_data, default=serialize_object)
    except TypeError as e:
        return f"Error serializing the request data: {str(e)}"

    result = llm.invoke(request_json)
    return result.content
