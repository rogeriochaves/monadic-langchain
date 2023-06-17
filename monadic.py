from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
)
import langchain
import langchain.schema

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# Monadic Chain


def identity_parser(x: str):
    return x


class Chain(Generic[T, U]):
    llm: BaseLanguageModel
    prompt: str
    input_mapper: Optional[Callable[[T], Dict[str, str]]]
    output_parser: Callable[[str], U]
    named_class: Type[langchain.LLMChain]

    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        prompt: str,
        input_mapper: Optional[Callable[[T], Dict[str, str]]] = None,
        output_parser: Callable[[str], U] = identity_parser,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.input_mapper = input_mapper
        self.output_parser = output_parser
        self.named_class = type(name, (langchain.LLMChain,), {})

    def call(self, input: T, callbacks: Callbacks = []) -> U:
        input_values = self.input_mapper(input) if self.input_mapper else input
        if type(input_values) is not dict:
            raise Exception("cannot extract input_values")

        prompt = langchain.PromptTemplate(
            template=self.prompt,
            input_variables=[str(k) for k in input_values.keys()],
        )
        chain = self.named_class(llm=self.llm, prompt=prompt, callbacks=callbacks)

        result = chain.run(input_values)
        result = self.output_parser(result)
        return result

    def __call__(self, input: T, callbacks: Callbacks = []) -> U:
        return self.call(input, callbacks)

    def and_then(self, fn: Callable[[U], "Chain[U, V]"]):
        return PipeChain(chain_a=self, chain_b_fn=fn)


class IdentityChain(Chain[T, T]):
    def __init__(self) -> None:
        pass

    def call(self, input: T, _callbacks: Callbacks = []):
        return input


class ConstantChain(Chain[Any, U]):
    output: U

    def __init__(self, output: U) -> None:
        self.output = output

    def call(self, _input: Any, _callbacks: Callbacks = []):
        return self.output


class PipeChain(Chain[T, V]):
    chain_a: Chain[T, Any]
    chain_b_fn: Callable[[Any], Chain[Any, V]]

    def __init__(
        self, chain_a: Chain[T, U], chain_b_fn: Callable[[U], Chain[Any, V]]
    ) -> None:
        self.chain_a = chain_a
        self.chain_b_fn = chain_b_fn

    def call(self, input: T, callbacks: Callbacks = []):
        output = self.chain_a.call(input, callbacks)
        chain_b = self.chain_b_fn(output)
        return chain_b.call(output, callbacks)
