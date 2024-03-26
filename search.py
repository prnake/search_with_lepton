import concurrent.futures
import json
import re
import os
import traceback
import threading
from typing import Annotated, List, Generator, Optional

import httpx
import requests
from loguru import logger

BACKEND_MODEL = "gpt-4-turbo-preview"
# BACKEND_MODEL = "glm-4"
# BACKEND_MODEL = "mixtral-8x7b-32768"

SERPER_API = os.environ["SERPER_API"]

SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"

# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5

# If the user did not provide a query, we will use this default query.
_default_query = "What is the answer to life, the universe, and everything?"

# This is really the most important part of the rag model. It gives instructions
# to the model on how to generate the answer. Of course, different models may
# behave differently, and we haven't tuned the prompt to make it optimal - this
# is left to you, application creators, as an open problem.
_rag_query_text = """
You are a large language AI assistant built by Lepton AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# This is the prompt that asks the model to generate related questions to the
# original question and the contexts.
# Ideally, one want to include both the original question and the answer from the
# model, but we are not doing that here: if we need to wait for the answer, then
# the generation of the related questions will usually have to start only after
# the whole answer is generated. This creates a noticeable delay in the response
# time. As a result, and as you will see in the code, we will be sending out two
# consecutive requests to the model: one for the answer, and one for the related
# questions. This is not ideal, but it is a good tradeoff between response time
# and quality.
_more_questions_prompt = """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Use line breaks to separate different questions. Here is the original question:
"""

# fastapi api settings
from pydantic import BaseModel
from typing import Optional
class QueryRequest(BaseModel):
    query: str 
    search_uuid: str
    generate_related_questions: Optional[bool] = True

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

# thread safe openai client
# openai client is currently a universal protocol
def local_openai_client():
    """
    Gets a thread-local client, so in case openai clients are not thread safe,
    each thread will have its own client.
    """
    import openai

    thread_local = threading.local()
    try:
        return thread_local.client
    except AttributeError:
        thread_local.client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_TOKEN"),
            timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
        )
        return thread_local.client


def search_with_serper(query: str, subscription_key: str):
    """
    Search with serper and return the contexts.
    """
    payload = json.dumps({
        "q": query,
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    })
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content["knowledgeGraph"].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append({
                    "name": json_content["knowledgeGraph"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content["answerBox"].get("answer")
            if url and snippet:
                contexts.append({
                    "name": json_content["answerBox"].get("title",""),
                    "url": url,
                    "snippet": snippet
                })
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet","")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []

app = FastAPI()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

def get_related_questions(query, contexts):
    """
    Gets related questions based on the query and context.
    """

    try:
        response = local_openai_client().chat.completions.create(
            model=BACKEND_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _more_questions_prompt.format(
                        context="\n\n".join([c["snippet"] for c in contexts])
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            max_tokens=512,
        )
        related = response.choices[0].message.content.split("\n")
        return [{"question": q.strip()} for q in related if q.strip()]
    except Exception as e:
        # For any exceptions, we will just return an empty list.
        logger.error(
            "encountered error while generating related questions:"
            f" {e}\n{traceback.format_exc()}"
        )
        return []


def _raw_stream_response(
        contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            try:
                result = json.dumps(related_questions)
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

@app.post("/query")
async def query_function(body: QueryRequest) -> StreamingResponse:
    """
    Query the search engine and returns the response.

    The query can have the following fields:
        - query: the user query.
        - generate_related_questions: if set to false, will not generate related
            questions. Otherwise, will depend on the environment variable
            RELATED_QUESTIONS. Default: true.
    """
    query = body.query
    search_uuid = body.search_uuid
    generate_related_questions = body.generate_related_questions
    logger.info(f"Received query: {query}")
    logger.info(f"Received search_uuid: {search_uuid}")
    logger.info(f"Received generate_related_questions: {generate_related_questions}")
    query = query or _default_query
    # Basic attack protection: remove "[INST]" or "[/INST]" from the query
    query = re.sub(r"\[/?INST\]", "", query)

    contexts = search_with_serper(query, SERPER_API)

    system_prompt = _rag_query_text.format(
        context="\n\n".join(
            [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
        )
    )

    try:
        llm_response = local_openai_client().chat.completions.create(
            model=BACKEND_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,
            stream=True,
            temperature=0.9,
        )
        if generate_related_questions:
            # While the answer is being generated, we can start generating
            # related questions as a future.
            related_questions_future = executor.submit(
                get_related_questions, query, contexts
            )
        else:
            related_questions_future = None
    except Exception as e:
        logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
        return HTMLResponse("Internal server error.", 503)

    return StreamingResponse(
        _raw_stream_response(contexts, llm_response, related_questions_future),
        media_type="text/html",
    )

app.mount("/ui", StaticFiles(directory="ui"), name="static")

@app.get("/")
async def index(request: Request):
    """
    Redirects "/" to the ui page.
    """
    return RedirectResponse(url="/ui/index.html")

if __name__ == "__main__":
    import uvicorn
    logger.info("Running LLM Server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)