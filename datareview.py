# from openai import OpenAI
# import os

# def call_llm(prompt, model="mistral.mistral-7b-instruct-v0:2"):
#     """
#     Function to call OpenAI's LLM with a given prompt
    
#     Args:
#         prompt (str): The input text to send to the LLM
#         model (str): The model to use (default: gpt-3.5-turbo)
        
#     Returns:
#         str: The LLM's response
#     """
#     # Set your API key
#     # You can set it as an environment variable for security
#     #api_key = os.environ.get("OPENAI_API_KEY")
    
#     # If not set as environment variable, you can uncomment and add your key here
#     api_key = "8i3WJVpeVc6pVfNYD5ebRamQ64wIbMYXaAANMxxa"
    
#     try:
#         # Initialize the OpenAI client
#         client = OpenAI(api_key=api_key)
        
#         # Call the OpenAI API
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=500,
#             temperature=0.7
#         )
        
#         # Extract and return the response text
#         return response.choices[0].message.content.strip()
    
#     except Exception as e:
#         return f"Error calling the LLM: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     user_prompt = input("Enter your question for the LLM: ")
#     response = call_llm(user_prompt)
#     print("\nLLM Response:")
#     print(response)

import requests
import os
from typing import Dict, List, Optional, Union, Any

class CapgeminiGenAI:
    """
    Client for interacting with Capgemini's Generative AI Engine
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.capgemini-genai.com/v1"):
        """
        Initialize the Capgemini GenAI client
        
        Args:
            api_key (str, optional): API key for authentication. If not provided, will look for CAPGEMINI_API_KEY env var
            base_url (str): Base URL for the Capgemini GenAI API
        """
        self.api_key = api_key or os.environ.get("CAPGEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via CAPGEMINI_API_KEY environment variable")
        
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
    def generate_text(self, 
                     prompt: str, 
                     model: str = "capgemini-gpt-default", 
                     max_tokens: int = 500,
                     temperature: float = 0.7,
                     top_p: float = 1.0,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate text using Capgemini's Generative AI model
        
        Args:
            prompt (str): The input text prompt
            model (str): The model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dict[str, Any]: The API response containing generated text
        """
        endpoint = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Capgemini GenAI API: {str(e)}")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "capgemini-chat-default",
                       max_tokens: int = 500,
                       temperature: float = 0.7,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate chat completions using Capgemini's Generative AI model
        
        Args:
            messages (List[Dict[str, str]]): List of message objects with role and content
            model (str): The model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dict[str, Any]: The API response containing the chat completion
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Capgemini GenAI API: {str(e)}")
    
    def embeddings(self, 
                  input_text: Union[str, List[str]], 
                  model: str = "capgemini-embedding-default") -> Dict[str, Any]:
        """
        Generate embeddings for the given input text
        
        Args:
            input_text (Union[str, List[str]]): Text to generate embeddings for
            model (str): The embedding model to use
            
        Returns:
            Dict[str, Any]: The API response containing the embeddings
        """
        endpoint = f"{self.base_url}/embeddings"
        
        payload = {
            "model": model,
            "input": input_text if isinstance(input_text, list) else [input_text]
        }
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Capgemini GenAI API: {str(e)}")


# Example usage
def main():
    # Set your API key
    api_key = "your_capgemini_api_key_here"  # Or set as environment variable
    
    # Initialize the client
    client = CapgeminiGenAI(api_key=api_key)
    
    # Example: Generate text
    text_response = client.generate_text(
        prompt="Explain the concept of generative AI in simple terms.",
        max_tokens=300
    )
    print("Text Generation Response:")
    print(text_response)
    print("\n" + "-"*50 + "\n")
    
    # Example: Chat completion
    chat_response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the key benefits of using generative AI in business?"}
        ]
    )
    print("Chat Completion Response:")
    print(chat_response)
    print("\n" + "-"*50 + "\n")
    
    # Example: Generate embeddings
    embedding_response = client.embeddings(
        input_text="Generative AI is transforming businesses."
    )
    print("Embedding Response:")
    print(embedding_response)


if __name__ == "__main__":
    main()