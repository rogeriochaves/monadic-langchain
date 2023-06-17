import re
from typing import (
    Literal,
    TypedDict,
    cast,
)
from dotenv import load_dotenv

from monadic import Chain, ConstantChain

load_dotenv()

import langchain
import langchain.schema

from langchain.chat_models import ChatOpenAI

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


routing_chain = Chain[str, RoutingChainOutput](
    "RoutingChain",
    llm=llm,
    prompt="""
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
    input_mapper=lambda input: {"question": input},
    output_parser=lambda output: {
        "action": cast(
            Literal["SEARCH", "REPLY"], simple_key_extract("Action", output)
        ),
        "param": simple_key_extract("Param", output),
    },
)

summarizer_chain = Chain[str, str](
    "SummarizerChain",
    llm=llm,
    prompt="Summarize the following text: {text}\nSummary: ",
    input_mapper=lambda input: {"text": input},
)

search_chain = Chain[RoutingChainOutput, str](
    "SearchChain",
    llm=llm,
    prompt="Pretend to search for the user. Query: {query}\nResults: ",  # this would be replace with a proper vector db search
    input_mapper=lambda input: {"query": input["param"]},
).and_then(lambda _: summarizer_chain)

conversation_chain = routing_chain.and_then(
    lambda output: ConstantChain(output["param"])
    if output["action"] == "REPLY"
    else search_chain
)

import chainlit as cl


@cl.langchain_factory(use_async=False)
def factory():
    return conversation_chain
