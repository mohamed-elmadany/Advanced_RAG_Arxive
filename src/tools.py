import ollama

model = "qwen3.5:4b"

def weather(city):
    return f"The weather in {city} is sunny with a high of 25°C and a low of 15°C."
def get_current_time(city):
    return f"The current time in {city} is 3:00 PM."

tools=[
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather in a given city",
            "parameters":{
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city for which to get the weather"
                    }
                },
                "required": ["city"]
            }

        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a given city",
            "parameters":{
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city for which to get the current time"
                    }
                },
                "required": ["city"]
            }

        }
    }
]
def execute_tool(tool_name, parameters):
    tool_map={
        "weather": weather,
        "get_current_time": get_current_time
    }
    if tool_name in tool_map:
        return tool_map[tool_name](**parameters)

class llm:
    def __init__(self):
        self.model=model
    def ask(self, message):
        response = ollama.chat(model=self.model, messages=message,tools=tools, stream=True)
        full_response = ""
        tool_calls = []
        for word in response:
            if word.message.get("content"):
                print(word.message.content, end='', flush=True)
                full_response += word.message.content
                
            elif word.message.get("tool_calls"):
                tool_calls.extend(word.message.tool_calls)
                print(word.message.tool_calls)
        if full_response or tool_calls:
            message.append({'role': 'assistant', 'content': full_response, 'tool_calls': tool_calls})

        if tool_calls:
            for tool_call in tool_calls:
                tool_response = execute_tool(tool_call.function.name, tool_call.function.arguments)
                print(f"\nTool response: {tool_response}")
                message.append({"role": "tool","tool_name": tool_call.function.name ,"content": tool_response})
            return self.ask(message)


if __name__ == "__main__":
    llm_instance = llm()
    while True:
        user_input = input("\n>> ")
        if user_input.lower() == "/exit":
            print("Goodbye!")
            break
        message=[{"role": "user", "content": user_input}]
        llm_instance.ask(message)
