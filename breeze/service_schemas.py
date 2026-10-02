"""Small, explicitly supported chat-completion request contract."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=65536)

    @field_validator("content")
    @classmethod
    def content_is_plain_text(cls, value):
        if not value.strip():
            raise ValueError("Message content cannot be blank")
        if "<|" in value or "|>" in value:
            raise ValueError("Model control-token delimiters are not accepted in message content")
        if "\x00" in value:
            raise ValueError("NUL characters are not accepted")
        return value


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1, max_length=100)
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    temperature: Literal[0] = 0
    top_p: Literal[1] = 1
    n: Literal[1] = 1
    stream_options: StreamOptions | None = None

    @field_validator("temperature", "top_p", "n", mode="before")
    @classmethod
    def not_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("Numeric options cannot be booleans")
        return value

    @model_validator(mode="after")
    def conversation_contract(self):
        turns = self.messages[1:] if self.messages[0].role == "system" else self.messages
        if not turns or turns[-1].role != "user":
            raise ValueError("End the conversation with a user message")
        if any(message.role != ("user" if index % 2 == 0 else "assistant")
               for index, message in enumerate(turns)):
            raise ValueError("Use an optional first system message, then alternating user and assistant messages")
        if sum(len(message.content) for message in self.messages) > 65536:
            raise ValueError("Combined message content exceeds 65536 characters")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self


class ServiceError(Exception):
    def __init__(self, message, status=400, code="invalid_request", error_type="invalid_request_error"):
        super().__init__(message)
        self.message, self.status, self.code, self.error_type = message, status, code, error_type

    def payload(self, request_id):
        return {"error": {"message": self.message, "type": self.error_type,
                          "param": None, "code": self.code}, "request_id": request_id}