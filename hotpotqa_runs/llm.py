from typing import Union, Literal
from langchain.chat_models import ChatOpenAI
#from langchain import OpenAI
from openai import OpenAI
import os
from langchain.schema import (
    HumanMessage
)

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = "gpt-oss" # kwargs.get('model_name', 'gpt-3.5-turbo') 
        #if model_name.split('-')[0] == 'text':
        #*args, **kwargs,
        self.model = OpenAI(base_url = "https://ellm.nrp-nautilus.io/v1", api_key=os.environ['OPENAI_API_KEY'])
        self.model_type = 'completion'
        # else:
        #     self.model = ChatOpenAI(*args, **kwargs, base_url = "https://ellm.nrp-nautilus.io/v1", api_key=os.environ['OPENAI_API_KEY'])
        #     self.model_type = 'chat'
    
    def __call__(self, prompt: str,system_prompt: str):
        try:
            completion = self.model.chat.completions.create(
                model="gpt-oss",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            res= completion.choices[0].message.content
            # print(res)
            return res
            # if self.model_type == 'completion':
            #     return self.model(prompt)
            # else:
            #     return self.model(
            #         [
            #             HumanMessage(
            #                 content=prompt,
            #             )
            #         ]
            #     ).content
        except Exception as e:
            print(e)


            # +'The output format should be strictly following the examples.'