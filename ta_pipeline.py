from docx import Document
import re
from collections import defaultdict
from vector_utils import vectorize_text, match_quote_fast
import fitz
import math
#from google.colab import userdata
import gradio as gr
import time
from spaces import GPU
import model_utils
#from resources import model, tokenizer
from prompts import (
    code_prompt, 
    cluster_prompt, 
    summary_prompt, 
    chat_prompt,
    custom_user_prompt
)

chat_marker = "**Answer**"
code_marker = "===ANNOTATIONS START==="
cluster_marker = "Do not repeat codes within a theme.\n"
summary_marker = "\n---\n\n**"

## Load the User's Document
def update_upload_status(file_input):
    if not file_input or not hasattr(file_input, "name"):
        return gr.update(visible=True), "⚠️ No file detected. Please re-upload."
    else:
        file_path = file_input.name

    if file_path[-5:] == ".docx":
        return gr.update(visible=True), "DOCX file uploaded."

    elif file_path[-4:] == ".pdf":
        return gr.update(visible=True), "PDF file uploaded."

    elif file_path[-4:] == ".txt":
        return gr.update(visible=True), "TXT file uploaded."

    else:
        return gr.update(visible=True), "❌ Unsupported file type"


def load_doc(file_input, path=False):
    if not file_input:
        return None

    if path:
        file_path = file_input

    else:
        file_path = file_input.name

    if file_path[-5:] == ".docx":
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)

        text = "\n".join(text)

    elif file_path[-4:] == ".pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

    elif file_path[-4:] == ".txt":
        with open(file_path, "r") as file:
            text = file.read()

    elif file_path[-4:] == ".json":
        # Might add something here to let it handle JSON files
        pass

    else:
        return None

    return text


@GPU
def chat_with_text(question: str, text: str, chat_chain, already_vectorized: bool = False, threshold: float=0.5) -> list:
    if not already_vectorized:
        chunks, index, embeddings, embedder = vectorize_text(text)
    
    context = match_quote_fast(question, chunks, index, embeddings, embedder, threshold = 0.1)[0]

    if context is None:
        output = chat_chain.invoke({"question": question, 
                                    "context": "No relevant information about the user's question was found in the text."})
        return [output, None]
    else:
        output = chat_chain.invoke({"question": question, "context": context})

        return [output, question]


def parse_chat(answer: str, marker: str=chat_marker) -> str:
    idx = answer.rfind(marker)
    output = answer[idx+len(marker)+1:]
    output = output.strip()

    return output 


@GPU
def open_chat(model, tokenizer):
    llm = model_utils.make_llm(model, tokenizer, temperature=0)
    return llm, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


def handle_chat(question, text, llm):
    chat_chain = chat_prompt | llm
    reply = chat_with_text(question, text, chat_chain)
    return parse_chat(reply[0])


def chunk_text_by_tokens(text, tokenizer, max_tokens=1024, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunk_text = tokenizer.decode(chunk)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # add overlap for context

    return chunks


@GPU
def code_text(text, tokenizer, code_chain, n_codes=-1, marker=code_marker, chunk_size=1024, user_prompt='', batch_size=6, progress=gr.Progress()):
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=chunk_size)
    n_chunks = len(chunks)
    all_outputs = []

    prompt_template = code_chain.first
    llm = code_chain.last

    # Validate prompt
    required_inputs = prompt_template.input_variables
    use_user_prompt = "user_prompt" in required_inputs
    use_n_codes = "n_codes" in required_inputs

    # Chunk text into token-limited slices
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=chunk_size)
    n_chunks = len(chunks)
    all_outputs = []
    n_batches = math.ceil(n_chunks / batch_size)
    j = 0

    for batch_start in range(0, n_chunks, batch_size):
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        prompts = []
        progress(j / n_batches, desc=f"Coding batch {j + 1} / {n_batches}...")
        #print(f"Coding batch {j + 1} / {n_batches}...")

        for chunk in batch_chunks:
            variables = {"text": chunk}
            if use_n_codes:
                variables["n_codes"] = n_codes if n_codes > 0 else "unlimited"
            if use_user_prompt:
                variables["user_prompt"] = user_prompt
            prompts.append(prompt_template.format(**variables))

        # Batched call to Hugging Face pipeline
        responses = llm.pipeline(prompts)

        for i, res in enumerate(responses):
            # Case 1: res is string
            if isinstance(res, str):
                output_text = res

            # Case 2: res is list of dicts
            elif isinstance(res, list) and isinstance(res[0], dict) and "generated_text" in res[0]:
                output_text = res[0]["generated_text"]

            # Case 3: res is dict
            elif isinstance(res, dict) and "generated_text" in res:
                output_text = res["generated_text"]

            else:
                raise ValueError(f"Unexpected response format: {res}")
            first = output_text.find(marker)
            second = output_text.find(marker, first + len(marker))
            clean_output = output_text[second:] if second != -1 else output_text[first:]
            new_output = {"chunk": batch_start + i, "output": clean_output}
            all_outputs.append(new_output)
            yield new_output  # for real-time feedback

        time.sleep(0.25)
        j += 1


def parse_codes(codes: list[dict], text: str) -> dict:
    temp_dict = defaultdict(list)

    pattern = r"^\s*\d+\.\s*(.+?)\s*\|\s*(?:<code>)?(.+?)(?:</code>)?\s*\|\s*(\d+)\s*$"

    for chunk in codes:
        lines = chunk['output'].strip().splitlines()
        for line in lines:
            matches = re.findall(pattern, line)
            for sentence, code, score in matches:
                code = code.title().replace("_", " ").replace(" Ai ", " AI ").replace(" Ai.", " AI.").strip()
                if (sentence, int(score)) not in temp_dict[code]:
                    temp_dict[code].append((sentence, int(score)))

    # resolve or remove hallucinations
    possible_hallucinations = []
    for code in list(temp_dict.keys()):
        for i, (sentence, conf) in enumerate(temp_dict[code]):
            if sentence not in text:
                possible_hallucinations.append({"code": code, "index": i, "sentence": sentence, "conf": conf})

    if possible_hallucinations:
        print(f"Found {len(possible_hallucinations)} possible hallucinations. Searching for actual quotes...")
        chunks, index, embeddings, embedder = vectorize_text(text)
        resolved = 0

        for h in possible_hallucinations:
            quote = match_quote_fast(h["sentence"], chunks, index, embeddings, embedder, threshold=0.65)
            del temp_dict[h["code"]][h["index"]]
            if quote[0]:
                temp_dict[h["code"]].append((quote[0], h["conf"]))
                resolved += 1
            else:
                if not temp_dict[h["code"]]:
                    del temp_dict[h["code"]]
        print(f"Was able to resolve {resolved} hallucinations.")

    return dict(temp_dict)


@GPU
def develop_themes(codes: str, tokenizer, cluster_chain, marker: str=cluster_marker, max_themes: int=-1, chunk_size:int=1024, progress=gr.Progress()) -> list:
    chunks = chunk_text_by_tokens(codes, tokenizer, max_tokens=chunk_size)
    n_chunks = len(chunks)
    all_outputs = []
    for i, chunk in enumerate(chunks):
        #print(f"Clustering into themes... chunk {i+1}/{len(chunks)}...")
        progress(i/n_chunks, desc=f"Clustering into themes... chunk {i+1}/{len(chunks)}...")
        if max_themes > 0:
          output = cluster_chain.invoke({"codes": chunk, "max_themes": max_themes})
        else:
          output = cluster_chain.invoke({"codes": chunk, "max_themes": "unlimited"})
        idx = output.find(cluster_marker)
        all_outputs.append({"chunk": i+1, "output": output[idx+len(cluster_marker):]})

        yield all_outputs
        time.sleep(0.5)


def parse_themes(themes: list[dict]) -> dict[str, list[str]]:
    theme_dict = defaultdict(list)
    current_theme = None

    for chunk in themes:
      for line in chunk['output'].strip().splitlines():
          line = line.strip()

          # Detect theme lines
          theme_match = re.match(r"^Theme:\s*(.+)", line, re.IGNORECASE)
          if theme_match:
              current_theme = theme_match.group(1).strip()
              continue

          # Detect codes
          code_match = re.match(r"^Codes:\s*(.+)", line, re.IGNORECASE)
          if code_match and current_theme:
              codes = [code.strip() for code in code_match.group(1).split("|")]
              for code in codes:
                  if code not in theme_dict[current_theme]: # avoid redundancy
                      code = code.title()
                      code = code.replace("_"," ")
                      code = code.replace(" Ai ", " AI ").replace(" Ai.", " AI.")
                      theme_dict[current_theme].append(code)

    return dict(theme_dict)


@GPU
def summarize_themes(theme_dict: list[dict], text: str, tokenizer, summary_chain, marker: str=summary_marker, chunk_size: int=1024, progress=gr.Progress()) -> list:
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=chunk_size)
    n_chunks = len(chunks)
    all_outputs = []
    for i, chunk in enumerate(chunks):
        #print(f"Summarizing themes... chunk {i+1}/{len(chunks)}...")
        progress(i/n_chunks, desc=f"Summarizing themes... chunk {i+1}/{len(chunks)}...")
        output = summary_chain.invoke({"themes": theme_dict, "text": chunk})
        idx = output.find(marker)
        all_outputs.append({"chunk": i+1, "output": output[idx+len(marker):]})

        # check for cancellations
        yield all_outputs
        time.sleep(0.5)


def parse_summaries(summaries: list) -> list[dict]:
    # want to extract in format {theme: summary}
    summary_dict = {}

    # Summarization was run for each chunk of text.
    for chunk in summaries:
      theme_blocks = re.split(r"Theme\*\*:", chunk['output'].strip())

      for block in theme_blocks[1:]:  # skip anything before the first theme
          lines = block.strip().splitlines()

          # remove white space and asterisks
          theme = lines[0].strip()
          theme = re.match(r'[^*]+', theme).group(0)

          # add the theme if it is not already in the dictionary
          if theme not in summary_dict:
              summary_dict[theme] = {"summary": ""}

          summary_line = next((l for l in lines if l.startswith("**Summary**:")), None)

          if not summary_line:
              continue  # skip incomplete blocks

          summary = summary_line.replace("**Summary**:", "").strip()

          # include the summary if it is not already there (first summary for a theme will be used)
          if len(summary_dict[theme]["summary"]) == 0:
              summary_dict[theme]["summary"] = summary

    return summary_dict


def combine(theme_dict, summary_dict, code_dict) -> list[dict]:
    combined_dict = {}
    for theme in theme_dict.keys():
      # make a list with two dictionaries for each theme
      try:
          combined_dict[theme] = [summary_dict[theme],{}]
      except KeyError:
          continue

      for code in theme_dict[theme]:
        try:
          combined_dict[theme][1][code] = code_dict[code]
        except KeyError:
          # do not include any codes that are cut off
          continue

    return combined_dict