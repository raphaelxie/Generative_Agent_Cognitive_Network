"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.

NOTE (migration patch):
- This file has been updated to use the OpenAI Python SDK (openai>=1.0).
- Old calls like openai.ChatCompletion.create / openai.Completion.create / openai.Embedding.create
  have been replaced with client.chat.completions.create and client.embeddings.create.
- API key is read from environment variable OPENAI_API_KEY.
- Default chat model can be set via environment variable OPENAI_MODEL.
"""

import json
import time
import os

from openai import OpenAI

# Keep utils import if other code relies on it.
# (If utils.py defines other constants/helpers used elsewhere, leave this.)
from utils import *

import pathlib, datetime

RUN_TAG = os.environ.get("GA_RUN_TAG", "run")
LOG_DIR = pathlib.Path("runs") / RUN_TAG
LOG_DIR.mkdir(parents=True, exist_ok=True)
LLM_LOG = LOG_DIR / "llm_calls.jsonl"

# ----------------------------------------------------------------------------
# Global client + defaults
# ----------------------------------------------------------------------------

# Reads API key from environment variable. You already verified this works:
#   export OPENAI_API_KEY="..."
_api_key = os.environ.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=_api_key)

# Default model can be overridden:
#   export OPENAI_MODEL="gpt-5.2-mini"  (or any model available to your account)
DEFAULT_CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Embedding model (keep stable for reproducibility)
DEFAULT_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def temp_sleep(seconds: float = 0.1):
    time.sleep(seconds)


# ----------------------------------------------------------------------------
# Internal helper: minimal chat wrapper
# ----------------------------------------------------------------------------

def _chat(prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = None) -> str:
    temp_sleep()
    used_model = (model or DEFAULT_CHAT_MODEL)

    resp = client.chat.completions.create(
        model=used_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""

    # minimal logging
    record = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "type": "chat",
        "model": used_model,
        "prompt_chars": len(prompt),
        "output_chars": len(text),
        "run_tag": RUN_TAG,
    }
    with open(LLM_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return text



# ----------------------------------------------------------------------------
# SECTION 1: CHATGPT-* STRUCTURE (kept for backward compatibility)
# ----------------------------------------------------------------------------

def ChatGPT_single_request(prompt: str) -> str:
    # Historically used for quick one-off requests
    return _chat(prompt)


def GPT4_request(prompt: str) -> str:
    """
    Legacy name kept for compatibility with the original codebase.
    Uses DEFAULT_CHAT_MODEL unless overridden by OPENAI_MODEL.
    """
    try:
        return _chat(prompt)
    except Exception as e:
        print("ChatGPT ERROR:", e)
        return "ChatGPT ERROR"


def ChatGPT_request(prompt: str) -> str:
    """
    Legacy name kept for compatibility with the original codebase.
    Uses DEFAULT_CHAT_MODEL unless overridden by OPENAI_MODEL.
    """
    try:
        return _chat(prompt)
    except Exception as e:
        print("ChatGPT ERROR:", e)
        return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt,
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False):
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate and func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            pass

    return False


def ChatGPT_safe_generate_response(prompt,
                                  example_output,
                                  special_instruction,
                                  repeat=3,
                                  fail_safe_response="error",
                                  func_validate=None,
                                  func_clean_up=None,
                                  verbose=False):
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate and func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response

            if verbose:
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")

        except Exception:
            pass

    return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                      repeat=3,
                                      fail_safe_response="error",
                                      func_validate=None,
                                      func_clean_up=None,
                                      verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate and func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")
        except Exception:
            pass

    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


# ----------------------------------------------------------------------------
# SECTION 2: ORIGINAL GPT-3 STRUCTURE (migrated)
# ----------------------------------------------------------------------------

def GPT_request(prompt, gpt_parameter):
    """
    Backward-compatible wrapper.

    Original code used openai.Completion.create with an 'engine' (e.g. gpt-5.2).
    The modern SDK uses chat.completions. We keep the interface and map:
      - gpt_parameter["engine"] -> model
      - gpt_parameter["temperature"] -> temperature
      - gpt_parameter["max_tokens"] -> max_tokens

    If gpt_parameter refers to a deprecated model, this may fail; in that case,
    set OPENAI_MODEL to a valid model and/or update gpt_parameter["engine"].
    """
    temp_sleep()
    try:
        model = gpt_parameter.get("engine", DEFAULT_CHAT_MODEL)
        temperature = gpt_parameter.get("temperature", 0.7)
        max_tokens = gpt_parameter.get("max_tokens", None)
        return _chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return "error"


def generate_prompt(curr_input, prompt_lib_file):
    """
    Loads a prompt template and substitutes !<INPUT k>! placeholders.
    """
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    with open(prompt_lib_file, "r") as f:
        prompt = f.read()

    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)

    marker = "<commentblockmarker>###</commentblockmarker>"
    if marker in prompt:
        prompt = prompt.split(marker)[1]

    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if curr_gpt_response == "error":
            continue
        if func_validate and func_validate(curr_gpt_response, prompt=prompt):
            return func_clean_up(curr_gpt_response, prompt=prompt) if func_clean_up else curr_gpt_response
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")

    return fail_safe_response


# ----------------------------------------------------------------------------
# Embeddings (migrated)
# ----------------------------------------------------------------------------

def get_embedding(text: str, model: str = None):
    text = text.replace("\n", " ")
    used_model = model or DEFAULT_EMBED_MODEL

    resp = client.embeddings.create(
        model=used_model,
        input=text
    )

    embedding = resp.data[0].embedding

    # logging
    record = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "type": "embed",
        "model": used_model,
        "input_chars": len(text),
        "output_dim": len(embedding),
        "run_tag": RUN_TAG,
    }

    with open(LLM_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return embedding


# ----------------------------------------------------------------------------
# Local test
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    gpt_parameter = {
        "engine": DEFAULT_CHAT_MODEL,  # replace old DEFAULT_CHAT_MODEL
        "max_tokens": 50,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ['"']
    }

    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response, prompt=None):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True

    def __func_clean_up(gpt_response, prompt=None):
        return gpt_response.strip()

    output = safe_generate_response(
        prompt,
        gpt_parameter,
        5,
        "rest",
        __func_validate,
        __func_clean_up,
        True
    )

    print(output)
