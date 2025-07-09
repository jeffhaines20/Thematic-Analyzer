---
title: Thematic Analyzer
emoji: 🚀
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
short_description: 'Conducts thematic analysis of documents using an LLM. '
---
# 🧠 Thematic Analysis with LLMs

This Hugging Face Space allows you to upload a document and automatically extract, code, and visualize themes using an LLM (default is LLaMA 3 8B). The app supports multi-code tagging, theme-to-color mapping, and interactive tooltips with blended highlighting for overlapping themes.

## 🚀 Features

- 🗂 Upload `.docx`, `.pdf`, or `.txt` files
- 🤖 Automatic coding using a fine-tuned LLM (LLaMA 3 or other)
- 🎨 Inline HTML text highlighting with:
  - Code tooltips
  - Text colored by theme with theme tooltips
  - Multi-theme gradient spans
- 📊 Visualizations of:
  - Code and theme networks
  - Word clouds
  - Frequency plots
- 🔎 Customizable prompt injection and LLM settings

## 🧪 Example Workflow

1. Upload a document via the file uploader.
2. Select desired LLM parameters (temperature, chunk size, max tokens).
3. Click **🏷️ Code Text**.
4. Review highlighted HTML output with coded text.
5. Click **📚 Cluster Codes into Themes**.
6. Review highlighted HTML output with theme-aware spans and theme network graph.
7. Click **📝 Summarize Themes**.
8. Review and download output table, and visual analytics (sankey plot, word cloud).
9. Use optional features like Named-Entity Extraction and Run Aggregation for further analysis.

## 📁 File Structure

.  
├── assets  
&nbsp;&nbsp;&nbsp;&nbsp;      ├── Examples            # Text and dictionaries to load as examples for demonstration or troubleshooting  
&nbsp;&nbsp;&nbsp;&nbsp;      └── Experiment Results  # Dataframe and dictionaries hold results of experiment runs with different parameter values  
├── app.py                  # Gradio interface entry point  
├── doc_utils.py            # Loading and downloading html   
├── model_utils.py          # LLM loading and wrappers  
├── ner_utils.py            # Working with Named Entity Recognition  
├── prompts.py              # Coding, theme clustering, and summarizing prompts  
├── resources.py            # Holder for model and tokenizer  
├── run_management.py       # Functions for aggregating multiple LLM outputs  
├── ta_pipeline.py          # Text loading, preprocessing, chunking, llm calls, and chat functions   
├── vector_utils.py         # Vectorizing the text for retrieval augmented generation  
├── viz.py                  # Text highlighting, network graphs, sankey chart, word clouds, aggregation visualizations  
├── requirements.txt        # Dependencies  
└── README.md               # This file 

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
