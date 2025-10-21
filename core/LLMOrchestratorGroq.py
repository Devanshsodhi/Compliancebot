# core/llm_orchestrator_groq.py
import requests
import os

class LLMOrchestratorGroq:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "mixtral-8x7b"  # You can change to "llama3-70b" etc.
        self.available = self.api_key is not None

    def _call_groq(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def answer_question(self, question, context_results):
        """Answer a question using retrieved context"""
        context = "\n\n".join([c['text'] for c in context_results])
        messages = [
            {"role": "system", "content": "You are a compliance and document analysis assistant."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
        ]
        return self._call_groq(messages)

    def compliance_analysis(self, document, rules):
        """Run compliance analysis on a single document"""
        messages = [
            {"role": "system", "content": "You are an expert compliance analyst."},
            {"role": "user", "content": f"Analyze this document for compliance based on the following rules:\n\n{rules}\n\nDocument:\n{document}"}
        ]
        return self._call_groq(messages)
