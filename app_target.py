import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace

class ModelEndpoints:
    def __init__(self: Self, env: dict, model_type: str) -> str:
        self.env = env
        self.model_type = model_type
    
    def __call__(self: Self, question: str) -> Response:
        if self.model_type == "gpt4-0613":
            output = self.call_gpt4_endpoint(question)
        elif self.model_type == "gpt35-turbo":
            output = self.call_gpt35_turbo_endpoint(question)
        elif self.model_type == "mistral7b":
            output = self.call_mistral_endpoint(question)
        elif self.model_type == "tiny_llama":
            output = self.call_tiny_llama_endpoint(question)
        elif self.model_type == "phi3_mini_serverless":
            output = self.call_phi3_mini_serverless_endpoint(question)
        elif self.model_type == "gpt2":
            output = self.call_gpt2_endpoint(question)
        else:
            output = self.call_default_endpoint(question)
        return output
    
    def query (self: Self, endpoint: str, headers: str, payload: str) -> str:
        response = requests.post(url=endpoint, headers=headers, json=payload)
        return response.json()
    
    def call_gpt4_endpoint(self: Self, question: str) -> Response:
        endpoint = self.env["gpt4-0613"]["endpoint"]
        key = self.env["gpt4-0613"]["key"]
        headers = {
            "Content-Type": "application/json",
            "api-key": key
            }
        payload = {
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 500,       
            }
        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        answer = output["choices"][0]["message"]["content"]
        return {"question": question, "answer": answer}
    
    def call_gpt4o_endpoint(self: Self, question: str) -> Response:
        endpoint = self.env["gpt4o"]["endpoint"]
        key = self.env["gpt4o"]["key"]
        headers = {
            "Content-Type": "application/json",
            "api-key": key
            }
        payload = {
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 500,       
            }
        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        answer = output["choices"][0]["message"]["content"]
        return {"question": question, "answer": answer}