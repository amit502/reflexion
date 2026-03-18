import os
import sys
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

openai.api_key = os.getenv('OPENAI_API_KEY')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model='gpt-oss', temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     max_tokens=max_tokens,
    #     stop=stop_strs,
    #     temperature=temperature,
    # )
    # return response.choices[0]["message"]["content"]
    model = OpenAI(base_url = "https://ellm.nrp-nautilus.io/v1", api_key=os.environ['OPENAI_API_KEY'])
    try:
        completion = model.chat.completions.create(
            model="gpt-oss",
            temperature=0.0,
            #stop=stop_strs,
            messages=messages
        )
        res= completion.choices[0].message.content
        if stop_strs and '\n' in res:
            for line in res.split('\n'):
                line = line.strip()
                if line:
                    res = line
                    break
            res = res.lstrip('>').strip()
        print(res)
        return res
    except Exception as e:
        print(e)
        return ""
