import pandas as pd
import numpy as np
import time
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pipelines import pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
import gc
import pickle
import gradio as gr
from datetime import datetime
from transformers.pipelines.text2text_generation import ReturnType
from spaces import GPU

# Local Dependencies
import viz
from run_management import update_run_selector
import ta_pipeline as ta
from prompts import (
    code_prompt,
    cluster_prompt,
    summary_prompt,
    chat_prompt,
    custom_user_prompt
)

model = None
tokenizer = None
model_choices = {
        "LLaMA 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "LLaMA 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
        "LLaMa 4 Scout Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "DeepSeek 7B Chat": "deepseek-ai/deepseek-llm-7b-chat"
    }

def build_model(use_llm, model_name):
    # Model options
    global model, tokenizer
   
    model_id = model_choices[model_name]

    if use_llm == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            use_safetensors=True,
            token=True,
            revision="main"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=True,
            revision="main"
        )

        return f"✅ {model_id} model is loaded."

    else:
        return f"No model was loaded because use_llm = False."


@GPU
def update_model(use_llm, model_name):
    # Model options
    global model, tokenizer

    model = None
    tokenizer = None
    torch.cuda.empty_cache()
    gc.collect()
    
    model_id = model_choices[model_name]

    if use_llm == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            use_safetensors=True,
            token=True,
            revision="main"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=True,
            revision="main"
        )

        return f"✅ {model_id} model is loaded."

    else:
        return f"No model was loaded because use_llm = False."


def make_llm(model, tokenizer, temperature=0, token_limit=-1): 
    if token_limit < 1:
      generate_kwargs = {
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.01
      } 

    else:
      generate_kwargs = {
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.01,
        "max_new_tokens": token_limit
      } 

    pipe = pipeline(
      "text-generation",
      model = model,
      tokenizer = tokenizer,
      **generate_kwargs
    )

    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':temperature})

    return llm


def log_model_performance(new_row: dict):
    token = os.getenv("HF_WRITE_TOKEN")
    repo_id = "jeffhaines/thematic-analysis-model-performance-log"

    # Try to load existing dataset
    try:
        ds = load_dataset(repo_id, split="train", use_auth_token=token)
        df = ds.to_pandas()
    except Exception as e:
        print("⚠️ Couldn't load dataset. Starting fresh.")
        df = pd.DataFrame()

    # Append the new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Convert to HF dataset and push
    updated_ds = Dataset.from_pandas(df)
    updated_ds.push_to_hub(repo_id, token=token)
    print("✅ Model performance row pushed to dataset.")


@GPU(duration=120)
def code(selected_model, file_input, n_codes=0, temperature=0, user_prompt='', use_example=False, session_runs=[], token_limit=-1, chunk_size=1024, track_performance=False):
    global model, tokenizer

    batch_size = 1
    # record start time to evaluate performance
    start = time.perf_counter()

    if session_runs is None:
        session_runs = []

    # use saved dictionary to conserve resources
    if use_example:
        try:
            with open("assets/Examples/tony_code_dict.pickle", "rb") as f:
                code_dict = pickle.load(f)
        except Exception as e:
            yield None, None, session_runs, f"❌ Error loading example code data: {str(e)}", None, None, None, None
            return

        try:
            full_text = ta.load_doc("assets/Examples/Tony Avidnote.docx", path=True)
        except Exception as e:
            yield None, None, session_runs, f"❌ Error loading example text: {str(e)}", None, None, None, None
            return

        try:
           highlighted_html = viz.generate_highlighted_html(full_text, code_dict)
        except Exception as e:
            yield None, None, session_runs, f"❌ Error highlighting html: {str(e)}", None, None, None, None
            return

        coding_result = {
        "label": f"Type: Codes - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
        "code_dict": code_dict
        }

        session_runs.append(coding_result)
        coding_status = "✅ Successfully loaded coding example."
        code_table = viz.code_dict_to_df(code_dict)

        run_selector_update = update_run_selector(session_runs)

        yield highlighted_html, code_dict, session_runs, coding_status, run_selector_update, run_selector_update, None, code_table
        return

    # otherwise make llm, chain, and run prompt
    if file_input is None:
        yield None, None, session_runs, "Please upload a file.", None, None, None, None
        return
    elif type(file_input) == str:
        full_text = ta.load_doc(file_input, path=True)
    else:
        full_text = ta.load_doc(file_input)

    n_codes_corrected = int(n_codes * 0.75) #align user-given number of codes with codes per chunk given to llm
    
    llm = make_llm(model, tokenizer, temperature=temperature, token_limit=token_limit)

    code_chain = (custom_user_prompt if len(user_prompt) > 0 else code_prompt) | llm

    try:
        codes = []

        def collect_codes():
            for output in ta.code_text(full_text, tokenizer, code_chain, n_codes=n_codes_corrected, chunk_size=chunk_size, user_prompt=user_prompt, batch_size=batch_size):
                codes.append(output)
                yield None, None, session_runs, None, None, None, codes, None

        yield from collect_codes()
        # use this if cancellation is not needed: codes = ta.code_text(full_text, tokenizer, code_chain, n_codes=n_codes, chunk_size=chunk_size, user_prompt=user_prompt)
    
    except Exception as e:
        return None, None, session_runs, f"❌ Error in coding using LLaMA: {str(e)}", None, None, None, None

    try:
        code_dict = ta.parse_codes(codes, full_text)
    except Exception as e:
        return None, None, session_runs, f"❌ Error in parsing codes: {str(e)}. Codes: {codes}", None, None, None, None


    # save the results locally
    coding_result = {
        "label": f"Type: Codes - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
        "code_dict": code_dict
    }

    session_runs.append(coding_result)
    coding_status = "✅ Successfully coded text."
    highlighted_html = viz.generate_highlighted_html(full_text, code_dict)
    code_table = viz.code_dict_to_df(code_dict)

    # need to update the run selector
    run_selector_update = update_run_selector(session_runs)

    # record the performance of the model
    end = time.perf_counter()
    if track_performance == True:
        distinct_quotes = set([tup[0] for code in code_dict.keys() for tup in code_dict[code]])
        distinct_quotes = "".join(distinct_quotes)
        text_length = len(full_text)
        perc_covered = (len(distinct_quotes)/text_length) * 100
 
        new_row = {"New Tokens": token_limit,
               "Chunk Size": chunk_size,
               "N Codes": n_codes,
               "Codes Generated": len(code_dict.keys()),
               "Percentage of Text Covered": perc_covered,
               "Time (Seconds)": end - start,
               "Text Length": text_length,
               "Characters per Second": text_length/(end-start),
               "Date and Time": datetime.now().strftime('%H:%M:%S'),
               "User Prompt": user_prompt,
               "Model": selected_model,
               "Batch Size": batch_size
               }

        log_model_performance(new_row)

    yield highlighted_html, code_dict, session_runs, coding_status, run_selector_update, run_selector_update, codes, code_table


@GPU(duration=120)
def cluster(full_text, code_dict, max_themes, temperature, use_example, session_runs, token_limit, chunk_size):
    # needs to run the cluster_chain on the code_dict, return a theme_dict, and visualize the clusters
    global model, tokenizer

    if session_runs is None:
        session_runs = []
    
    # use saved dictionary to conserve resources
    if use_example:
        try:
            with open("assets/Examples/tony_theme_dict.pickle", "rb") as f:
                theme_dict = pickle.load(f)
                # make network graph
                theme_network_html = viz.visualize_theme_network_for_gradio(code_dict, theme_dict)

                # color-code text by theme
                html_highlighted_by_theme = viz.highlight_text_by_theme(full_text=full_text,
                    code_dict=code_dict,
                    theme_dict=theme_dict)
        except Exception as e:
            yield None, None, None, session_runs, f"❌ Error loading example theme data: {str(e)}", None, None
            return

        # save the results locally
        clustering_result = {
            "label": f"Type: Themes - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
            "theme_dict": theme_dict
        }

        session_runs.append(clustering_result)
        theme_status = "✅ Successfully loaded example themes."
        run_selector_update = update_run_selector(session_runs)

        yield theme_dict, theme_network_html, html_highlighted_by_theme, session_runs, theme_status, run_selector_update, run_selector_update
        return

    # error handling
    if code_dict is None:
        yield None, None, None, session_runs, f"❌ Error: You must code the text first.", None, None
        return
    elif len(code_dict.keys()) == 0:
        yield None, None, None, session_runs, f"❌ Error: You must code the text first.", None, None
        return

    llm = make_llm(model, tokenizer, temperature = temperature, token_limit=token_limit)
    cluster_chain = cluster_prompt | llm

    try:
        themes = []

        def collect_themes():
            for output in ta.develop_themes("|".join([key for key in code_dict.keys()]), tokenizer, cluster_chain, max_themes = max_themes, chunk_size=chunk_size):
                themes.extend(output)
                yield None, themes, session_runs, None, None, None, None

        yield from collect_themes()
        #themes = ta.develop_themes("|".join([key for key in code_dict.keys()]), tokenizer, cluster_chain, max_themes = max_themes, chunk_size=chunk_size)
    
    except Exception as e:
        yield None, None, None, session_runs, f"❌ Error in clustering using LLaMA: {str(e)}.", None, None
        return
    try:
        theme_dict = ta.parse_themes(themes)
    except Exception as e:
        yield None, None, None, session_runs, f"❌ Error in parsing themes: {str(e)}.", None, None
        return

    # save the results locally
    clustering_result = {
        "label": f"Type: Themes - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
        "theme_dict": theme_dict
        }

    session_runs.append(clustering_result)
    theme_status = "✅ Successfully clustered codes into themes."

    # make network graph
    theme_network_html = viz.visualize_theme_network_for_gradio(code_dict, theme_dict)

    # color-code text by theme
    html_highlighted_by_theme = viz.highlight_text_by_theme(full_text=full_text,
        code_dict=code_dict,
        theme_dict=theme_dict)

    # need to update the run selectors
    run_selector_update = update_run_selector(session_runs)

    yield theme_dict, theme_network_html, html_highlighted_by_theme, session_runs, theme_status, run_selector_update, run_selector_update


@GPU(duration=120)
def summarize(theme_dict, code_dict, text, temperature, use_example, session_runs, token_limit, chunk_size):
    # needs to run the summary_chain on theme_dict, then combine all dictionaries and return a table of the combined_dict
    global model, tokenizer
    
    if session_runs is None:
        session_runs = []

    # use saved dictionary to conserve resources
    if use_example:
        try:
            with open("assets/Examples/tony_combined_dict.pickle", "rb") as f:
                combined_dict = pickle.load(f)
        except Exception as e:
            yield None, None, session_runs, f"❌ Error loading example combined data: {str(e)}", None, None
            return

        theme_tag_df = viz.dict_to_table(combined_dict)
        theme_df_html = viz.render_dataframe_as_html(theme_tag_df)

        session_result = {
          "label": f"Type: Total - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
          "combined_dict": combined_dict
          }

        session_runs.append(session_result)
        summarizing_status = "✅ Successfully loaded example summaries."

        # need to update the run selectors
        run_selector_update = update_run_selector(session_runs)

        yield combined_dict, theme_df_html, session_runs, summarizing_status, run_selector_update, run_selector_update
        return

    # error handling
    if code_dict is None:
        yield None, None, session_runs, f"❌ Error: You must code the text first.", None, None
        return

    elif len(code_dict.keys()) == 0:
        yield None, None, session_runs, f"❌ Error: You must code the text first.", None, None
        return

    if theme_dict is None:
        yield None, None, session_runs, f"❌ Error: You must cluster themes first.", None, None 
        return

    elif len(theme_dict.keys()) == 0:
        yield None, None, session_runs, f"❌ Error: You must cluster themes first.", None, None
        return

    llm = make_llm(model, tokenizer, temperature = temperature, token_limit=token_limit)
    summary_chain = summary_prompt | llm

    try:
        summaries = []

        def collect_summaries():
            for output in ta.summarize_themes(theme_dict, text, tokenizer, summary_chain, chunk_size=chunk_size):
                summaries.extend(output)
                yield None, summaries, session_runs, None, None, None

        yield from collect_summaries()
        #summaries = ta.summarize_themes(theme_dict, text, tokenizer, summary_chain, chunk_size=chunk_size)
    
    except Exception as e:
        yield None, None, session_runs, f"❌ Error in summarizing using LLaMA: {str(e)}.", None, None
        return

    try:
        summary_dict = ta.parse_summaries(summaries)
    except Exception as e:
        yield None, None, session_runs, f"❌ Error in parsing summaries: {str(e)}.", None, None
        return

    combined_dict = ta.combine(theme_dict, summary_dict, code_dict)
    theme_tag_df = viz.dict_to_table(combined_dict)
    theme_df_html = viz.render_dataframe_as_html(theme_tag_df)

    session_result = {
        "label": f"Type: Total - Run {len(session_runs)+1} - {datetime.now().strftime('%H:%M:%S')}",
        "combined_dict": combined_dict
        }

    session_runs.append(session_result)
    summarizing_status = "✅ Successfully summarized themes."

    # need to update the run selectors
    run_selector_update = update_run_selector(session_runs)

    yield combined_dict, theme_df_html, session_runs, summarizing_status, run_selector_update, run_selector_update

