from typing import Any, Optional
import sys

import streamlit as st

from langchain_core.callbacks import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler,BaseCallbackManager


class StreamingCallbackHandler(StdOutCallbackHandler):
  def __init__(self, color: Optional[str] = None):
    super().__init__(color)
    self.token_area = st.empty()
    self.tokens_stream = ""
  def on_llm_new_token(self, token: str, **kwargs: Any) -> str:
    '''新しいtokenが来たらprintする'''
    self.tokens_stream += token
    self.token_area.markdown(self.tokens_stream)
    print(token)
    # sys.stdout.write(token)
    return token

