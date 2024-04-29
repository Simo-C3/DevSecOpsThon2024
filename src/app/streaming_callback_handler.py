from typing import Any, Optional,Dict, List
import sys

import streamlit as st

from langchain_core.callbacks import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler,BaseCallbackManager


class StreamingCallbackHandler(StdOutCallbackHandler):
  def __init__(self, color: Optional[str] = None):
    super().__init__(color)
    self.token_area = None
    self.tokens_stream = ""
  def on_llm_new_token(self, token: str, **kwargs: Any) -> str:
    self.tokens_stream += token
    response = self.token_area.markdown(self.tokens_stream)

    # print(token)
    return token
  def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
    # print("Chain started")
    self.token_area = st.chat_message("assistant").empty()
  def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
    # print("LLM started")
    self.token_area = st.chat_message("assistant").empty()
  def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
    print("LLM ended")
    # print(response)
