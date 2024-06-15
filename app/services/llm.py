import os
from typing import List, Tuple

from langchain_community.llms.ollama import Ollama
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser, BaseTransformOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from loguru import logger


class BaseChatModel:
    def __init__(self, name: str, api_key: str | None = None):
        self.name = (name,)
        self.api_key = api_key

    def is_api_key_needed(self) -> bool:
        return self.api_key is not None

    def get_available_models(self) -> List[Tuple]:
        raise NotImplementedError

    def invoke(self, model_name: str, prompt: str, labels: dict = None) -> BaseMessage:
        raise NotImplementedError

    def _get_model(self, model_name: str) -> Runnable:
        raise NotImplementedError

    def chain(
        self,
        prompt: ChatPromptTemplate,
        model_name: str,
        output_parser: BaseTransformOutputParser = StrOutputParser(),
    ):
        return prompt | self._get_model(model_name) | output_parser


class OllamaChatModel(BaseChatModel):
    def __init__(self):
        super().__init__("Ollama")

    def get_available_models(self) -> List[Tuple]:
        llama3 = ("1", "llama3:latest", "")
        phi3 = ("2", "phi3:medium", "")
        gemma7b = ("3", "gemma:7b", "")

        return [llama3, phi3, gemma7b]

    def invoke(self, model_name: str, prompt: str, labels=None) -> BaseMessage:
        if labels is None:
            labels = {}
        llm = Ollama(model=model_name)
        result = llm.invoke(prompt, labels=labels)
        return BaseMessage(content=result, type="str")


class NvidiaFoundationChatModel(BaseChatModel):
    def __init__(self, api_key: str):
        super().__init__("Nvidia Foundation", api_key=api_key)
        os.environ["NVIDIA_API_KEY"] = api_key

    def get_available_models(self) -> List[Tuple]:
        try:
            supported_models = [
                "meta/llama3-70b-instruct",
                "microsoft/phi-3-medium-4k-instruct",
                "google/gemma-7b",
            ]
            models = ChatNVIDIA.get_available_models()
            return [
                (m.id, m.model_name, m.path)
                for m in models
                if m.model_name in supported_models
            ]
        except Exception as e:
            logger.opt(exception=e).error("the models could not be loaded")
            return []

    def _get_model(self, model_name: str) -> Runnable:
        return ChatNVIDIA(model=model_name)

    def invoke(self, model_name: str, prompt: str, labels: dict = None) -> BaseMessage:
        llm = ChatNVIDIA(model=model_name)
        result = llm.invoke(prompt)
        return result
