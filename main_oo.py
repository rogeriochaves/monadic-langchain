import re
from typing import Any, Dict, Literal, TypedDict, cast
from dotenv import load_dotenv

load_dotenv()

import langchain
import langchain.schema

from langchain.chat_models import ChatOpenAI
from langchain import (
    LLMChain,
    PromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain.callbacks.manager import (
    Callbacks,
)
import chainlit as cl

# Setup

langchain.debug = True

llm = ChatOpenAI(client=None, model="gpt-3.5-turbo", temperature=0)


def simple_key_extract(key: str, output: str) -> str:
    found = re.search(f"{key}: (.*)", output)
    if found is None:
        raise Exception("Parsing Error")

    return found[1]


# Example


class RoutingChainOutput(TypedDict):
    action: Literal["SEARCH", "REPLY"]
    param: str


class RoutingParser(BaseOutputParser[RoutingChainOutput]):
    def parse(self, output: str) -> Dict[str, Any]:
        return {
            "action": cast(
                Literal["SEARCH", "REPLY"], simple_key_extract("Action", output)
            ),
            "param": simple_key_extract("Param", output),
        }


class RoutingChain(LLMChain):
    def __init__(self):
        return super().__init__(
            llm=llm,
            prompt=PromptTemplate(
                template="""
                    You are a chatbot that helps users search on the documentation, but you can also do some chit-chatting with them.
                    Choose the action REPLY if the user is just chit-chatting, like greeting, asking how are you, etc, but choose SEARCH \
                    for everything else, so you can actually do the search and help them.

                    =============================

                    Input: hello there
                    Action: REPLY
                    Param: hey there, what are you looking for?

                    Input: how does langchain work?
                    Action: SEARCH
                    Param: langchain how it works

                    Input: code example of vector db
                    Action: SEARCH
                    Param: vector db code example

                    Input: how is it going?
                    Action: REPLY
                    Param: I'm going well, how about you?

                    Input: {question}
                """,
                input_variables=["question"],
                output_parser=RoutingParser(),
            ),
        )


class SummarizerChain(LLMChain):
    def __init__(self):
        return super().__init__(
            llm=llm,
            prompt=PromptTemplate(
                template="Summarize the following text: {text}\nSummary:",
                input_variables=["text"],
            ),
        )


class SearchChain(LLMChain):
    def __init__(self):
        return super().__init__(
            llm=llm,
            prompt=PromptTemplate(
                template="Pretend to search for the user. Query: {query}\nResults: ",
                input_variables=["query"],
            ),
        )


def conversation(input: str, callbacks: Callbacks) -> str:
    route = cast(
        RoutingChainOutput,
        RoutingChain().predict_and_parse(callbacks=callbacks, question=input),
    )
    if route["action"] == "REPLY":
        return route["param"]
    elif route["action"] == "SEARCH":
        result = SearchChain().__call__({"query": route["param"]}, callbacks=callbacks)
        result = SummarizerChain().__call__(
            {"text": result["text"]}, callbacks=callbacks
        )

        return result["text"]
    else:
        return f"unknown action {route['action']}"


@cl.langchain_factory(use_async=False)
def factory():
    return conversation
