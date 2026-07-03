import os
from dotenv import load_dotenv
load_dotenv()


def setup_langsmith():
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "autoinsight")
        print("[LangSmith] Tracing enabled.")
    else:
        print("[LangSmith] Tracing disabled.")
    return tracing