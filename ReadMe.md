# This is simple Agentic AI use case:
- In this project I am using arXiv, wikipedia, and DuckDuckGo agents
- Any query will be answered based on the data from these agents


# Env variables::
- Using streamlit secrets, add the following (any of the three is mandatory):
    ```toml
    [OpenAI]
    API_KEY = "your_openai_api_key"


    [Groq]
    API_KEY = "your_groq_api_key"


    [Google]
    API_KEY = "your_google_api_key"
    ```


# Future Plans:
- To add more agents in it and expand it to:
    + Google Search
    + Weather Search
    + YouTube Search
    + ...

