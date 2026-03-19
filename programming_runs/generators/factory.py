from .model import ModelBase, GPT4, GPT35, GPTDavinci, GptOss, StarChat, CodeLlama


def generator_factory(lang: str):
    if lang == "py":
        from .py_generate import PyGenerator
        return PyGenerator()
    elif lang == "rs":
        from .rs_generate import RsGenerator
        return RsGenerator()
    else:
        raise ValueError(f"Invalid language: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "gpt-oss":
        return GptOss()
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        version = model_name.split("-")[-1]
        return CodeLlama(version)
    elif model_name == "text-davinci-003":
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")