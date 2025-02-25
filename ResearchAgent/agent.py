"""
Chat with Agent for testing

JCA
"""

from time import time
import json
import datetime

import transformers
import torch


class Agent():
    def __init__(self, model_path, initial_message, save_path):
        # load_in_4bit=True
        self.pipeline = transformers.pipeline(
            "text-generation",
            model = model_path,
            model_kwargs = {"torch_dtype": torch.bfloat16},
            device_map = "auto",
        )
        self.model_path = model_path

        self.save_path = save_path

        # Initialize the conversation history
        self.initial_message = initial_message
        self.messages = [
            initial_message,
        ]

        # Used for save the chat history
        # In a more easy to read format
        self.chat_history = []

    def ask_llm(self):
        """Ask LLm model and print response"""
        init = time()
        response = self.pipeline(self.messages)
        model_response = response[0]["generated_text"][-1]['content']
        self.messages.append({"role": "assistant", "content": model_response})

        elapsed = round(time()-init, 1)
        response = f"\n ** Assistant ({elapsed}s) **: {model_response}"
        self.chat_history.append(response)
        print(response)


    def save_chat(self):
        timestamp =  '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        filename = f'{self.save_path}/chat-{timestamp}.txt'

        record = [
            f'MODEL_ID: {self.model_path} \n',
            f'INITIAL_MESSAGE: {self.initial_message} \n',
            '\n ----------- \n'
            ]

        record += self.chat_history

        with open(filename, 'w') as f:
            for line in record:
                f.write(line)


    def chat_with_model(self):
        """interact with the model in a chat-like manner"""
        print("Welcome to the chat! Type 'exit' to quit.")

        # Pass first message to LLM
        self.ask_llm(self.messages)

        while True:
            # Get user input
            user_input = input("\n ** You ** : ")
            if user_input.lower() == "exit":
                print("Exiting the chat. Goodbye!")
                break

            # Add the user's input to the message history
            self.messages.append({"role": "user", "content": user_input})
            self.chat_history.append(f'\n **You** : {user_input}')

            self.ask_llm()


    

