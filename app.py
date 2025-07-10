## libraries
import gradio as gr
import numpy as np
import pickle
from huggingface_hub import login
import os
import torch

# import local dependencies
import viz
import ta_pipeline as ta
import ner_utils as ner
import run_management as rm
import doc_utils as du
import model_utils
from config import use_llama, model_path

# Model options
model_choices = {
    "LLaMA 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "LLaMA 3.3": "meta-llama/Meta-Llama-3-8B-v3.3",
    "LLaMa 4 Scout Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "DeepSeek 7B Chat": "deepseek-ai/deepseek-llm-7b-chat"
}

login(token=os.environ["HUGGINGFACE_TOKEN"])

# Global variables
model = None
tokenizer = None

def code_wrapper(file_input, n_codes, temperature, user_prompt, use_example, session_runs, token_limit, chunk_size):
    return model_utils.code(model, tokenizer, file_input, n_codes, temperature, user_prompt, use_example, session_runs, token_limit, chunk_size)

def cluster_wrapper(full_text, code_dict_state, max_themes, temperature, use_example, session_runs, token_limit, chunk_size):
    return model_utils.cluster(model, tokenizer, full_text, code_dict_state, max_themes, temperature, use_example, session_runs, token_limit, chunk_size)

def summarize_wrapper(theme_dict_state, code_dict_state, full_text, temperature, use_example, session_runs, token_limit, chunk_size):
    return model_utils.summarize(model, tokenizer, theme_dict_state, code_dict_state, full_text, temperature, use_example, session_runs, token_limit, chunk_size)

def chat_wrapper():
    return ta.open_chat(model, tokenizer)

model, tokenizer, initial_model_status = model_utils.build_model(True, "LLaMA 3.1 8B")
model.to("cuda")


    
# Gradio Interface
with gr.Blocks(title="LLaMA 3 Thematic Analyzer") as demo:
    gr.Markdown("## 🧠 Thematic Analyzer")

    # objects
    llm_state = gr.State()
    code_dict_state = gr.State()
    theme_dict_state = gr.State()
    combined_dict_state = gr.State()
    summary_dict_state = gr.State()
    session_runs = gr.State([])  # save each run for future use

    with gr.Accordion("❓ Help", open=False):
        gr.Markdown('''The general workflow of the app is to
        
        1. set model and chunking parameters if desired
        
        2. load a document
        
        3. use Llama to code the document
        
        4. cluster the codes into themes
        
        5. summarize the themes. 
        
        Beyond that there are options to identify entities (proper nouns) in the text, 
        chat with the text, and, if you have coded, clustered,
        or summarized multiple times, aggregate and analyze the results.''')

    with gr.Tabs():
        with gr.Tab("Model and Text Chunking Parameters"):
            with gr.Row():
                use_llm = gr.Checkbox(label="Use an LLM", value=True, visible=False)
                selected_model = gr.Radio(choices=list(model_choices.keys()), value="LLaMA 3.1 8B", label="Select Model")
                load_model_button = gr.Button("Load Selected Model")
                gr.HTML("<span title='Load the selected model. Note that some models may take several minutes to load.'>ℹ️</span>")
                model_load_status = gr.Textbox(label="Model Status", value="❌ No model loaded.")   
                #print("DEBUG: Inside app.py tokenizer is", tokenizer_state)
                #print("DEBUG: Inside app.py tokenizer.eos_token_id is", getattr(tokenizer_state, "eos_token_id", "None"))
                
                load_model_button.click(
                  fn=model_utils.build_model,
                  inputs=[use_llm, selected_model],
                  outputs=[model_load_status]
                )

            with gr.Row():
                token_limit = gr.Slider(0, 4096, step=200, value=0, label="Set maximum number of new tokens model can generate.")
                # note: I have found best results with setting a value of 1024 for the coding and clustering into themes and then 0 (default) for sumarizing
                gr.HTML("<span title='Set the maximum number of new tokens Llama can generate. A value of 0 allows Llama to use its default settings. Higher values may result in more detailed output, but increase runtime. I find that a value of 1024 usually allows for generating a large number of codes.'>ℹ️</span>")

            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="LLM Temperature")
                gr.HTML("<span title='Set the temperature of the model. Higher temperatures lead the model to be less deterministic and produce output with greater variation. Currently I only allow this to affect the coding stage and clustering and summarizing themes are done with temperature=0.'>ℹ️</span>")

            with gr.Row():
                chunk_size = gr.Slider(128,8192, step=200, value=1024, label="Set chunk size.")
                gr.HTML("<span title='Set how many tokens of your text the model will hold in its context window at once. Powers of two (e.g. 128, 256, 512,...) are recommended for optimal performance. If your document is smaller than the chunk size, the model will hold the entire document in its context window, but for large documents this may cause truncation. Lower chunk values results in more detailed output, but increase runtime.'>ℹ️</span>")

            with gr.Accordion("Experiment Results - Guidance for Picking Parameters", open=False):
                gr.Markdown('''Below are the results of experiments run with different settings for chunk sizes
                and maximum number of new tokens. These results are from using Llama 3.1 8B to code a sample text. In all cases the value for number of codes was set at 
                zero - in other words left up to the model, and no additional prompting was used. Performance may differ with specialized prompting. 
                These help to visualize the tradeoff between the time it takes to complete the coding and the comprehensiveness of the coding. In the first plot,
                there is a clear elbow where number of new tokens is 1024 and the chunk size is alsp 1024, indicating tht comprehensiveness
                slows down around here.''')
                plotly_output = gr.Plot()
                seaborn_output = gr.Plot()
                demo.load(viz.experimental_plots, outputs=[plotly_output, seaborn_output])
    
        with gr.Tab("📁 Load File"):
            use_example = gr.Checkbox(label="Use example coded data instead of LLM", value=False)
            
            with gr.Row():
                run_selector = gr.Dropdown(choices=[], label="Select a Past Run to Load", interactive=True)
                load_btn = gr.Button("Load Selected Run")
                gr.HTML("<span title='Load a previous run to visualize.'>ℹ️</span>")
                download_run_btn = gr.Button("📥 Download Selected Run", visible=True) # may do more with this in future
                gr.HTML("<span title='Download run data for for future use.'>ℹ️</span>")
                download_file = gr.File(label="Download Pickle File", visible=True)
            
            download_run_btn.click(
                fn=rm.download_selected_run,
                inputs=[run_selector, session_runs],
                outputs=[download_file])

            # to do
            #upload_run_file = gr.File(label="Upload Run (JSON)", file_types=[".json"])
            
            with gr.Row():
                file_input = gr.File(label="Upload Text (PDF, DOCX, or TXT)", file_types=[".pdf", ".docx", ".txt"])
                upload_status = gr.Textbox(label="Upload Status", visible=False)
            
            full_text = gr.Textbox(label="Original Text", lines=10, interactive=False)
            file_input.change(ta.load_doc, inputs=file_input, outputs=full_text).then(
              ta.update_upload_status, inputs=file_input, outputs=[upload_status, upload_status]
            )
            use_example.change(du.toggle_example_text, inputs=use_example, outputs=full_text)


        with gr.Tab("🏷️ Code Text"):
            with gr.Row():
                n_codes = gr.Slider(0, 20, value=5, step=1, label="Maximum Number of Codes per 1,000 Words")
                gr.HTML("<span title='Set the maximum number of codes you want the model to find per 1,000 words. A value of 0 sets no limit. This is a rough guide only, as Llama may or may not obey.'>ℹ️</span>")
                user_prompt = gr.Textbox(label="Try Your Own Prompt (Optional).")

            with gr.Row():
                code_button = gr.Button("🔎 Code Text")
                gr.HTML("<span title='LLaMA will read the document and code sentences it thinks are important. Use the sliders above to tune your parameters, and enter your own custom prompt if desired.'>ℹ️</span>")
                code_stop_button = gr.Button("Cancel")
                gr.HTML("<span title='Stop coding the text.'>ℹ️</span>")

            coding_status = gr.Textbox(label="Status", visible=True)
            with gr.Accordion("Raw Output (For Debugging)", open=False):
                raw_llm_output = gr.Textbox(label="Raw Output", lines=10)

            with gr.Accordion("📄 Coded Text", open=False):
                html_code_output = gr.HTML(label="Highlighted Text")
                #html_download_format = gr.Dropdown(["Word (.docx)", "PDF (.pdf)"], value=, label="Download Format")
                code_highlight_type = gr.State("Codes")
                download_coded_html_button = gr.Button("Download Highlighted Document")
                highlighted_code_output = gr.File(label="Click to Download")
                download_coded_html_button.click(
                    fn=du.export_html,
                    inputs=[html_code_output, code_highlight_type],
                    outputs=[highlighted_code_output])


        with gr.Tab("📚 Cluster Codes into Themes"):
            with gr.Row():
                max_themes = gr.Slider(0, 16, value=5, step=1, label="Maximum Number of Themes")
                gr.HTML("<span title='Set the maximum number of themes you want the model to find. A value of 0 sets no limit. This is a rough guide only, as Llama may or may not obey.'>ℹ️</span>")

            with gr.Row():
                cluster_button = gr.Button("📦 Cluster Themes")
                gr.HTML("<span title='LLaMA will take the code and cluster them into themes.'>ℹ️</span>")
                cluster_stop_button = gr.Button("Cancel")

            theme_status = gr.Textbox(label="Status", visible=True)

            with gr.Accordion("📚 Text Annotated By Theme", open=False):
                html_highlighted_by_theme = gr.HTML(label="Text Clustered by Theme")
                theme_highlight_type = gr.State("Themes")
                download_theme_html_button = gr.Button("Download Highlighted Document")
                highlighted_theme_output = gr.File(label="Click to Download")
                download_theme_html_button.click(
                    fn=du.export_html,
                    inputs=[html_highlighted_by_theme, theme_highlight_type],
                    outputs=[highlighted_theme_output])

            with gr.Accordion("🧩 Theme Clusters", open=False):
                theme_code_network_html = gr.HTML(label="Theme Network Visualization")
                network_highlight_type = gr.State("Network")
                download_networkcoded_html_button = gr.Button("Download Network Graph")
                theme_code_network_output = gr.File(label="Click to Download")
                download_coded_html_button.click(
                    fn=du.export_html,
                    inputs=[theme_code_network_html, network_highlight_type],
                    outputs=[theme_code_network_output])


        with gr.Tab("📝 Summarize Themes"):
            with gr.Row():
                summary_button = gr.Button("📝 Summarize Themes")
                gr.HTML("<span title='LLaMA will look at the themes and the text of the paper write a summary for each theme.'>ℹ️</span>")
                summary_stop_button = gr.Button("Cancel")

            summarizing_status = gr.Textbox(label="Status", visible=True)

            with gr.Accordion("🏷️ Theme Table", open=False):
                theme_df_html = gr.HTML(label="Theme Table")
                download_theme_table_button = gr.Button("Download Table")
                theme_table_output = gr.File(label="Click to download CSV")
                download_theme_table_button.click(
                    fn=du.download_theme_table,
                    inputs=[combined_dict_state],
                    outputs=[theme_table_output])

            with gr.Accordion("More Visualizations", open=False):                                
                with gr.Tab("📊 Sankey Plot"):
                    sankey_status = gr.Textbox(label="Status", visible=False)
                    sankey_button = gr.Button("📊 Visualize")
                    gr.HTML("<span title='Create a Sankey Graph.'>ℹ️</span>")
                    sankey_graph = gr.Plot(label="Sankey Diagram")
                    sankey_button.click(
                        fn=viz.visualize_sankey,
                        inputs=[combined_dict_state],
                        outputs=[sankey_graph, sankey_status, sankey_status])


                with gr.Tab("☁️ Word Cloud"): 
                    word_cloud_status = gr.Textbox(label="Status", visible=False)
                    make_cloud_button = gr.Button("📊 Visualize")
                    cloud_mode = gr.Radio(["Text", "Codes", "Quotes", "Themes"], label="Choose Wordcloud Mode", value="Text")
                    cloud_filter_type = gr.Dropdown(
                        choices=["None"], # make it so that different options are displayed depending on what mode is chosen
                        value="None",
                        label="Filter by...",
                        visible=True
                        )
                    cloud_filter_value = gr.Dropdown(
                        choices=["None"], # make it so that different options are displayed depending on what mode is chosen
                        value="None",
                        label="Filter Value",
                        visible=True
                        )

                    gr.HTML("<span title='Select whether to create a cloud from the entire original text, the parts of the text that were coded, or the codes.'>ℹ️</span>")
                                    
                    word_cloud = gr.Plot(label="Word Cloud of Non-Stop Words")

                    cloud_mode.change(
                        fn=rm.cloud_mode_change,
                        inputs=[cloud_mode],
                        outputs=[cloud_filter_type]
                    )

                    cloud_filter_type.change(
                        fn=rm.cloud_filter_type_change,
                        inputs=[combined_dict_state, cloud_filter_type],
                        outputs=[cloud_filter_value]
                    )

                    make_cloud_button.click(
                        fn=viz.make_word_cloud,
                        inputs=[combined_dict_state, full_text, cloud_mode, cloud_filter_type, cloud_filter_value],
                        outputs=[word_cloud, word_cloud_status, word_cloud_status])

            load_btn.click(
                fn=rm.load_selected_run,
                inputs=[full_text, run_selector, session_runs],
                outputs=[html_code_output,
                        code_dict_state,
                        theme_dict_state,
                        theme_code_network_html,
                        html_highlighted_by_theme,
                        combined_dict_state,
                        theme_df_html])

    #### Named Entity Recognition ####
    with gr.Accordion("📌 Named Entity Recognition", open=False):
        label_filter = gr.CheckboxGroup(
            choices=["PERSON", "ORG", "GPE", "LOC", "DATE", "PRODUCT", "EVENT"],
            label="Filter by Entity Type",
            value=["PERSON", "ORG", "GPE"]  # default selection
            )

        with gr.Row():
          ner_button = gr.Button("🔍 Tag Entities")
          gr.HTML("<span title='Use spaCY entity recognition on your text.'>ℹ️</span>")

        with gr.Tabs():
            with gr.Tab("Text"):
                ner_html_output = gr.HTML(label="Highlighted Output")
            with gr.Tab("Entity Table"):
                ner_output = gr.Dataframe(label="Named Entities", headers=["Entity", "Type"])

        ner_button.click(
            fn=ner.extract_named_entities,
            inputs=[full_text, label_filter],
            outputs=[ner_output, ner_html_output]
        )


    #### Chat with text using Retrieval Augmented Generation ####
    with gr.Accordion("💬 Chat with Text", open=False):
        with gr.Row():
            chat_button = gr.Button("💬 Chat with Text")
            gr.HTML("<span title='Chat with your text using Llama.'>ℹ️</span>")

        chatbot_box = gr.Chatbot(label="Chat Window", visible=False, type="messages")
        user_msg = gr.Textbox(label="Your Message", visible=False)
        chat_output = gr.Textbox(label="Response", lines=6)
        send_btn = gr.Button("Send", visible=False)

        chat_button.click(
            fn=chat_wrapper,
            inputs=[],
            outputs=[llm_state, chatbot_box, user_msg, send_btn])

        send_btn.click(
            fn=ta.handle_chat,
            inputs=[user_msg, full_text, llm_state],
            outputs=[chat_output]
        )

    #### Run Aggregation ####
    with gr.Accordion("🗃️ Run Aggregation", open=False):
        aggregated_dict_state = gr.State([])
        runs_to_aggregate = gr.State([])
        run_type_state = gr.State()

        # for debugging
        debug_button = gr.Button("🧪 Debug Session Runs", visible=False)
        debug_output = gr.JSON(label="Debug Output", visible=False)

        debug_button.click(
            lambda runs: runs,
            inputs=[aggregated_dict_state],
            outputs=[debug_output]
        )

        gr.HTML("<span title='The runs in this table will be aggegrated for analysis. This can be useful to uncover broader patterns that emerge from coding multiple times. Only runs of the same type can be aggregated.'>ℹ️</span>")
        add_status = gr.Textbox(label="Add Status", visible=False)

        run_view = gr.Dataframe(label="📋 Runs to Aggregate", interactive=False)
        status_box = gr.Textbox(label="Status", interactive=False, visible=False)
        selected_row_idx = gr.State(value=-1)

        # drop the run chosen from the table
        with gr.Row():
            drop_run_button = gr.Button("Drop Selected Run")
            drop_all_button = gr.Button("Drop All Runs")

        drop_run_button.click(
            fn=rm.drop_run,
            inputs=[selected_row_idx, runs_to_aggregate],
            outputs=[status_box, runs_to_aggregate]
        ).then(
        fn=lambda runs: rm.get_run_summary(runs),
        inputs=[runs_to_aggregate],
        outputs=[run_view]
        ).then(
            fn=rm.reset_index,
            outputs=selected_row_idx
        )

        drop_all_button.click(
            fn=rm.drop_all_runs,
            inputs=[runs_to_aggregate],
            outputs=[status_box, runs_to_aggregate]
        ).then(
            fn=lambda runs: rm.get_run_summary(runs),
            inputs=[runs_to_aggregate],
            outputs=[run_view]
        ).then(
            fn=lambda x: [],
            inputs=[aggregated_dict_state],
            outputs=[aggregated_dict_state]
        )

        run_view.select(
            fn=rm.select_row,
            inputs=[],
            outputs=[selected_row_idx]
        )

        with gr.Row():
            run_type_selector = gr.Dropdown(
                choices=["Codes", "Themes", "Total"],
                label="Select Type of Runs to Add",
                interactive=True
                )
            add_all_button = gr.Button("Add All Types of This Run")

        with gr.Row():
            available_runs = gr.Dropdown(choices=[], label="Select a Past Run to Add", interactive=True)
            add_button = gr.Button("➕ Add Run")

        add_button.click(
        fn=rm.add_run,
        inputs=[available_runs, session_runs, runs_to_aggregate],
        outputs=[runs_to_aggregate, add_status, add_status]
        ).then(
            fn=lambda runs: rm.get_run_summary(runs),
            inputs=[runs_to_aggregate],
            outputs=[run_view]
        )

        add_all_button.click(
            fn=rm.add_selected_run_type,
            inputs=[run_type_selector, session_runs, runs_to_aggregate],
            outputs=[runs_to_aggregate, add_status, add_status]
        ).then(
        fn=lambda runs: rm.get_run_summary(runs),
        inputs=[runs_to_aggregate],
        outputs=[run_view]
        )

        agg_status = gr.Markdown("Click the button to aggregate runs.")

        with gr.Row():
          aggregate_button = gr.Button("📊 Aggregate Selected Runs") ## write click method
          gr.HTML("<span title='The selected runs will be combined to be analyzed using visualizations in Aggregate Analysis.'>ℹ️</span>")

    #### Aggregation Analysis ####
    with gr.Accordion("🔎 Aggregate Analysis", open=False):
        with gr.Tabs():
            with gr.Tab("📊 Frequency Plots"):
                visualization_status = gr.Markdown("Click the button below to visualize aggregate data for your runs.")
                visualize_agg_button = gr.Button(" Aggregate Frequencies")
                code_freq_plot = gr.Plot(label="Code Frequency")
                code_quote_freq_plot = gr.Plot(label="Code-Quote Co-Occurrence")
                code_quote_data =gr.Dataframe(label="Code-Quote Table")
                theme_freq_plot = gr.Plot(label="Theme Frequency")
                theme_code_freq_plot = gr.Plot(label="Theme-Code Co-Occurrence")
                theme_code_data = gr.Dataframe(label="Theme-Code Table")

                visualize_agg_button.click(
                    fn=viz.visualize_aggregate,
                    inputs=[run_view, aggregated_dict_state],
                    outputs=[code_freq_plot,
                            code_quote_freq_plot,
                            code_quote_data,
                            theme_freq_plot,
                            theme_code_freq_plot,
                            theme_code_data,
                            visualization_status]
                )

            with gr.Tab("☁️ Word Clouds"):        
                agg_word_cloud_button = gr.Button("Make Word Cloud")
                source_selector = gr.Dropdown(
                    choices=[], # make it so that different options are displayed depending on what aggregate dictionary is loaded
                    label="Level to Visualize",
                    visible=True
                    )
                filter_type_selector = gr.Dropdown(
                    choices=["None"], # make it so that different option types are displayed depending on
                    value="None",
                    label="Filter by...",
                    interactive=True,
                    visible=True
                    )
                filter_value_selector = gr.Dropdown(
                    choices=["None"], # make it so that different option types are displayed depending on
                    value="None",
                    label="Select a value to filter for",
                    interactive=True,
                    visible=True
                    )
                agg_cloud_plot = gr.Plot(label="Word Cloud")

                # need to check the logic here and thoroughly test
                source_selector.change(
                    fn=rm.on_source_selector_change,
                    inputs=[source_selector, run_type_state],
                    outputs=[filter_type_selector, filter_value_selector]
                )

                filter_type_selector.change(
                    fn=rm.on_filter_type_change,
                    inputs=[source_selector, filter_type_selector, run_type_state, aggregated_dict_state],
                    outputs=[filter_value_selector]
                )

                agg_word_cloud_button.click(
                    fn=rm.make_agg_cloud,
                    inputs=[run_type_state, aggregated_dict_state, source_selector, filter_type_selector, filter_value_selector],
                    outputs=[agg_cloud_plot]
                )

        ## aggregate multiple runs
        aggregate_button.click(
            fn=rm.aggregate_selected_runs,
            inputs=[runs_to_aggregate],
            outputs=[aggregated_dict_state, source_selector, source_selector, run_type_state, agg_status]
        )


    # --- Core Button Logic ---
    code_event = code_button.click(
        fn=code_wrapper,
        inputs=[file_input, n_codes, temperature, user_prompt, use_example, session_runs, token_limit, chunk_size],
        outputs=[html_code_output, code_dict_state, session_runs, coding_status, run_selector, available_runs, raw_llm_output],
        )

    code_stop_button.click(fn=None, cancels=[code_event]).then(
      fn=lambda: gr.Textbox.update(value="❌ Coding cancelled."),
      inputs=None,
      outputs=coding_status
    )

    cluster_event = cluster_button.click(
            fn=cluster_wrapper,
            inputs=[full_text, code_dict_state, max_themes, temperature, use_example, session_runs, token_limit, chunk_size],
            outputs=[theme_dict_state, theme_code_network_html, html_highlighted_by_theme, session_runs, theme_status, run_selector, available_runs],
        )

    cluster_stop_button.click(fn=None, cancels=[cluster_event]).then(
      fn=lambda: gr.Textbox.update(value="❌ Clustering cancelled."),
      inputs=None,
      outputs=theme_status
    )

    summary_event = summary_button.click(
            fn=summarize_wrapper,
            inputs=[theme_dict_state, code_dict_state, full_text, temperature, use_example, session_runs, token_limit, chunk_size],
            outputs=[combined_dict_state, theme_df_html, session_runs, summarizing_status, run_selector, available_runs],
        )

    summary_stop_button.click(fn=None, cancels=[summary_event]).then(
      fn=lambda: gr.Textbox.update(value="❌ Summarizing cancelled."),
      inputs=None,
      outputs=summarizing_status
    )


demo.launch(ssr_mode=False)
