"""Help load models for local usage"""

from typing import Union
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

def process_model(model:str) -> Union[OpenAIChatModel, str]:
    """Process model and return `OpenAIChatModel` for local development with ollama
    
    Args:
        model: string in format {provider}:{model}:{model_version}

    Returns:
        str or OpenAIChatModel
    """

    if isinstance(model, OpenAIChatModel):
        return model
    provider, *model_name = model.split(":")
    model_name = ":".join(model_name)
    print(model_name)
    if provider == 'ollama':

        model = OpenAIChatModel(
            model_name=model_name,
            provider=OllamaProvider(base_url='http://localhost:11434/v1')
        )
    
    return model