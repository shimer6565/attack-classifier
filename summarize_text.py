import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI


def serialize_object(obj):
    """Custom serialization for objects that are not natively serializable by JSON."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def summarize_alert(event_json):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBET2W7lgxi2_600MBj9UQJ5KgQ5mUrcgk"

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    request_data = {
        "Event JSON": event_json,
        "task": "As an AI security analyst, review the provided JSON alert event. Identify and extract the name of the alert and a description of the events detailed in the JSON. Ensure that your response includes this information.",
    }

    try:
        request_json = json.dumps(request_data, default=serialize_object)
    except TypeError as e:
        return f"Error serializing the request data: {str(e)}"

    result = llm.invoke(request_json)
    return result.content
