import ollama
from core.config import config
 
class bot:
    def __init__(self):
        self.model=config.LLM_MODEL_NAME
        self.message_history = []
    def ask(self, message):
        self.message_history.append({"role": "user", "content": message})
        response = ollama.chat(model=self.model, messages=self.message_history, stream=True)
        full_response = ""
        for word in response:
            print(word.message.content, end='', flush=True)
            full_response += word.message.content
        assistant_message = {"role": "assistant", "content": full_response}
        self.message_history.append(assistant_message)
        return full_response
    def clear_context(self):
        self.message_history = []
    def chat(self):
        print("type '/bye' to exit")
        while True:
            message=input("\n>> ")
            if message == "/bye":
                break
            self.ask(message)
            
if __name__ == "__main__":
    llm = bot()
    llm.chat()
