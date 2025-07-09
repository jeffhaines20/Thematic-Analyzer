import viz
import gradio as gr
import pandas as pd
import tempfile
import pickle

def select_row(evt: gr.SelectData):
    row_index = evt.index[0]
    return row_index
    

def reset_index():
    return -1


def drop_all_runs(runs):
    runs = []
    return "✅ Removed all runs", runs


def make_code_and_theme_from_total(aggregated_combined_dict):
    code_dicts = []
    agg_theme_dict = {}
    for theme in aggregated_combined_dict.keys():
        agg_theme_dict[theme] = []
        for run in aggregated_combined_dict[theme]:
            code_dict = run[1]
            code_dicts.append(code_dict)
            agg_theme_dict[theme].extend([code for code in code_dict.keys()])

    agg_code_dict = {}
    for run in code_dicts:
        for code, tup in run.items():
            agg_code_dict.setdefault(code, []).extend(tup)

    return agg_code_dict, agg_theme_dict


def add_run(label, session_runs, runs_to_aggregate):
    # check to ensure all runs to aggregate are the same type
    run_type = label[6:11]
    if len(runs_to_aggregate) > 0:
        for run in runs_to_aggregate:
            if run["label"].startswith(f"Type: {run_type}"):
                continue
            else:
                return runs_to_aggregate, "❌ Error: Cannot combine multiple run types.", gr.update(visible=True)

    for r in session_runs:
        if r["label"] == label:
            runs_to_aggregate.append(r)

    return runs_to_aggregate, f"✅ Run added: {label}", gr.update(visible=True)


# this will add all runs of the selected run type to the runs_to_aggregate_list
def add_selected_run_type(run_type: str, session_runs: list[dict], runs_to_aggregate: list[dict]):
    filtered = [r for r in session_runs if r["label"].lower().startswith(f"type: {run_type.lower()}")]
    if len(runs_to_aggregate) > 0:
        for run in runs_to_aggregate:
            if run["label"].startswith(f"Type: {run_type}"):
                continue
            else:
                return runs_to_aggregate, "❌ Error: Cannot combine multiple run types.", gr.update(visible=True)

    for run in session_runs:
        if run["label"].startswith(f"Type: {run_type}"):
            runs_to_aggregate.append(run)

    return runs_to_aggregate, f"✅ {run_type} runs successfully added.", gr.update(visible=True)


def drop_run(index, runs):
    try:
        index = int(index)
        if index < 0:
            return "ℹ️ No action taken", runs
        label = runs[index]["label"]
        runs.pop(index)
        return f"✅ Removed: {label}", runs
    except Exception as e:
        return f"⚠️ Error removing run: {str(e)}", runs

    return "ℹ️ No action taken", runs


def load_selected_run(full_text, label, session_runs):
    for i, run in enumerate(session_runs):
        if run["label"] == label:
            if run["label"].startswith("Type: Codes"):
                code_dict = run["code_dict"]
                highlighted_html = viz.generate_highlighted_html(full_text, code_dict)

                return highlighted_html, code_dict, None, None, None, None, None

            elif run["label"].startswith("Type: Themes"):
                theme_dict = run["theme_dict"]

                # need to access the code dict of previous code run
                for j in range(i-1,-1,-1):
                    if session_runs[j]["label"].startswith("Type: Codes"):
                        code_dict = session_runs[j]["code_dict"]

                highlighted_html = viz.generate_highlighted_html(full_text, code_dict)
                theme_network_html = viz.visualize_theme_network_for_gradio(code_dict, theme_dict)
                # color-code text by theme
                html_highlighted_by_theme = viz.highlight_text_by_theme(full_text=full_text,
                    code_dict=code_dict,
                    theme_dict=theme_dict)

                return highlighted_html, code_dict, theme_dict, theme_network_html, html_highlighted_by_theme, None, None

            elif run["label"].startswith("Type: Total"):
                combined_dict = run["combined_dict"]
                code_dict = {code: tup for theme, entry in combined_dict.items() for code, tup in entry[1].items()}
                highlighted_html = viz.generate_highlighted_html(full_text, code_dict)
                theme_dict = {theme: [code for theme, entry in combined_dict.items() for code, tup in entry[1].items()] for theme in combined_dict.keys()}
                theme_network_html = viz.visualize_theme_network_for_gradio(code_dict, theme_dict)
                # color-code text by theme
                html_highlighted_by_theme = viz.highlight_text_by_theme(full_text=full_text,
                    code_dict=code_dict,
                    theme_dict=theme_dict)
                theme_tag_df = viz.dict_to_table(combined_dict)
                theme_df_html = viz.render_dataframe_as_html(theme_tag_df)

                return highlighted_html, code_dict, theme_dict, theme_network_html, html_highlighted_by_theme, combined_dict, theme_df_html


def get_run_summary(runs):
    df = pd.DataFrame([
        {
            "Label": run["label"],
            "Type": run["label"].split(':')[1].strip().split('-')[0]
        } for run in runs
    ])
    return df.astype(str)


## Aggregation
def aggregate_selected_runs(runs_to_aggregate):
    if len(runs_to_aggregate) > 0:
        run_type = runs_to_aggregate[0]["label"][6:11]
        if run_type == "Theme":
            run_type = "Themes"
        for run in runs_to_aggregate:
            if run["label"].startswith(f"Type: {run_type}"):
                continue
            else:
                return None, None, None, None, "❌ Error: Not all of the runs have the same type. Cannot aggregate."

    else:
        return None, None, None, None, "❌ Error: No runs to aggregate."

    if run_type == "Codes":
        # structure is {code: [list of (sentence, confidence) tuples]}
        aggregated = {}
        for run in runs_to_aggregate:
            for code, tup in run.get("code_dict", {}).items():
                aggregated.setdefault(code, []).extend(tup)

        dropdown_choices_1 = ["Quotes", "Codes"]

        return aggregated, gr.update(choices=dropdown_choices_1), gr.update(visible=True), run_type, "✅ Coding runs aggregated successfully!"

    elif run_type == "Themes":
        aggregated = {}
        for run in runs_to_aggregate:
            for theme, codes in run.get("theme_dict", {}).items():
                aggregated.setdefault(theme, []).extend(codes)

        dropdown_choices_1 = ["Codes", "Themes"]

        return aggregated, gr.update(choices=dropdown_choices_1), gr.update(visible=True), run_type, "✅ Theme runs aggregated successfully!"

    elif run_type == "Total":
        aggregated = {}
        for run in runs_to_aggregate:
            for theme, data in run.get("combined_dict", {}).items():
                # save each entry as a tuple (summary, {codes:(sentence, confidence)})
                aggregated.setdefault(theme, []).append(tuple(data))

        dropdown_choices_1 = ["Quotes", "Codes", "Themes"]


        return aggregated, gr.update(choices=dropdown_choices_1), gr.update(visible=True), run_type, "✅ Total runs aggregated successfully!"

    return None, None, None, None, "❌ Error: Unknown run type."


def update_run_selector(session_runs):
    options = [run["label"] for run in session_runs]
    if options:
        return gr.update(choices=options, value=options[-1])  # Set default to latest run
    else:
        return gr.update(choices=[], value=None)


def download_selected_run(label, session_runs):
    for run in session_runs:
        if run["label"] == label:
            # Save as a pickle file
            filename = label + ".pickle"
            dict_type = [key for key in run.keys()][1]
            with open(filename, "wb") as f:
                pickle.dump(run[dict_type], f)
                return filename

    return None


def get_theme_code_freqs(agg_theme_dict):
    theme_code_freqs = {}
    for theme in agg_theme_dict.keys():
        for code in agg_theme_dict[theme]:
            combined_key = f"{theme}: {code}"
            if combined_key not in theme_code_freqs:
                theme_code_freqs[combined_key] = 0
            theme_code_freqs[combined_key] += 1 # increase the frequency count

    return theme_code_freqs


def get_code_quote_freqs(agg_code_dict):
    code_quote_freqs = {}
    for code in agg_code_dict.keys():
        for tup in agg_code_dict[code]:
            quote = tup[0]
            combined_key = f"{code}: {quote}"
            if combined_key not in code_quote_freqs:
                code_quote_freqs[combined_key] = [0,0]
            code_quote_freqs[combined_key][0] += 1 # increase the frequency count
            code_quote_freqs[combined_key][1] += tup[1] # add the confidence

    for key in code_quote_freqs:
        count, total_conf = code_quote_freqs[key]
        code_quote_freqs[key][1] = total_conf / count  # get the average confidence

    return code_quote_freqs


def show_source_selector(aggregated_dict_state):
    # check if the aggregated dict exists and is empty
    if aggregated_dict_state is None:
        return gr.update(visible=False), "❌ Error: No runs to aggregate."
    elif len(aggregated_dict_state) == 0:
        return gr.update(visible=False), "❌ Error: No runs to aggregate."

    return gr.update(visible=True), "✅ Please select the type of content you want to build a word cloud from."


def cloud_mode_change(mode):
    if mode == "Text":
        return gr.update(choices=["None"], value="None")

    elif mode == "Codes":
        return gr.update(choices=["None", "Themes"], value="None")

    elif mode == "Quotes":
        return gr.update(choices=["None", "Codes", "Themes"])

    elif mode == "Themes":
        return gr.update(choices=["None"], value="None")

    else:
        # mode is None
        return gr.update(choices=["None"], value="None")


def cloud_filter_type_change(combined_dict, filter_type: str="None"):
    if filter_type == "None":
        return gr.update(choices=["None"], value="None")
    
    elif filter_type == "Codes":
        df = viz.dict_to_table(combined_dict)
        codes = [text for text in df['Code'].unique()]
        return gr.update(choices=["None"]+codes, value="None")

    elif filter_type == "Themes":
        df = viz.dict_to_table(combined_dict)
        themes = [text for text in df['Theme'].unique()]
        return gr.update(choices=["None"]+themes, value="None")


def on_source_selector_change(source, run_type_state):
    # user selects to make a word cloud from quotes when they have aggregated coding runs
    if source == "Quotes" and run_type_state == "Codes":
        return (
            gr.update(choices=["Codes"], value="Codes"),
            gr.update(choices=["None"], value="None")
        )

    # user selects to make a word cloud from codes when they have aggregated coding runs
    elif source == "Codes" and run_type_state == "Codes":
        return (
            gr.update(choices=["None"], value="None"),
            gr.update(choices=["None"], value="None") # quotes from all codes
        )

    # user selects to make a word cloud from codes when they have aggregated theme runs
    elif source == "Codes" and run_type_state == "Themes":
        return (
            gr.update(choices=["Themes"], value="Themes"),
            gr.update(choices=["None"], value="None")
        )

    # user selects to make a word cloud from themes when they have aggregated theme runs
    elif source == "Themes" and run_type_state == "Themes":
        return (
            gr.update(choices=["None"], value="None"),
            gr.update(choices=["None"], value="None")
        )

    # user selects to make a word cloud from quotes when they have aggregated total runs
    elif source == "Quotes" and run_type_state == "Total":
        return (
            gr.update(choices=["None","Codes","Themes"], value="None"),
            gr.update(choices=["None"], value="None")
        )

    # user selects to make a word cloud from codes when they have aggregated total runs
    elif source == "Codes" and run_type_state == "Total":
        return (
            gr.update(choices=["None","Themes"], value="None"),
            gr.update(choices=["None"], value="None")
        )

    # user selects to make a word cloud from themes when they have aggregated total runs
    elif source == "Themes" and run_type_state == "Total":
        return (
            gr.update(choices=["Themes"], value="Themes"),
            gr.update(choices=["None"], value="None")
        )

    else:
        return None, None


# need to set the logic so that if none is triggered here, the word cloud is made without going to the next level
def on_filter_type_change(source: str, filter_type: str, run_type_state: str, aggregated_dict_state):
    if source == "Quotes" and run_type_state == "Codes":
        return gr.update(choices=["None"] + list(aggregated_dict_state.keys()), value="None")

    elif source == "Codes" and run_type_state == "Themes":
        return gr.update(choices=["None"] + list(aggregated_dict_state.keys()), value="None")

    elif source == "Quotes" and run_type_state == "Total":
        agg_code_dict, agg_theme_dict = make_code_and_theme_from_total(aggregated_dict_state)
        if filter_type == "None":
            return gr.update(choices=["None"], value="None")
        elif filter_type == "Codes":
            return gr.update(choices=["None"] + list(agg_code_dict.keys()), value="None")
        elif filter_type == "Themes":
            return gr.update(choices=["None"] + list(agg_theme_dict.keys()), value="None")

    elif source == "Codes" and run_type_state == "Total":
        if filter_type == "None":
            return gr.update(choices=["None"], value="None")
        elif filter_type == "Themes":
            _, agg_theme_dict = make_code_and_theme_from_total(aggregated_dict_state)
            return gr.update(choices=["None"] + list(agg_theme_dict.keys()), value="None")

    else:
        return None


def make_agg_cloud(run_type_state: str, aggregated_dict_state, source: str, filter_type = "None", filter_value = "None"):
    # user selects to make a word cloud from quotes when they have aggregated coding runs
    if source == "Quotes" and run_type_state == "Codes":
        if filter_value != "None":
            # filter quotes by code
            text = " ".join([tup[0] for tup in aggregated_dict_state[filter_value]])
            title = f"Words Appearing in Quotes in Code {filter_value}"

        else: # do word cloud for all quotes
            text = " ".join([tup[0] for code in aggregated_dict_state.keys() for tup in aggregated_dict_state[code]])
            title = f"Words Appearing in Quotes"

        agg_cloud = viz.make_cloud(text, title)

        return agg_cloud

    # user selects to make a word cloud from codes when they have aggregated coding runs
    elif source == "Codes" and run_type_state == "Codes":
        text = " ".join([code for code in aggregated_dict_state.keys()])
        title = f"Words Appearing in Codes"
        agg_cloud = viz.make_cloud(text, title)
        return agg_cloud

    # user selects to make a word cloud from codes when they have aggregated theme runs
    elif source == "Codes" and run_type_state == "Themes":
        if filter_value != "None":
            # filter codes by theme
            text = " ".join([code for code in aggregated_dict_state[filter_value]])
            title = f"Words Appearing in Codes in Theme {filter_value}"
        else: # do word cloud for all quotes
            text = " ".join([code for theme in aggregated_dict_state.keys() for code in aggregated_dict_state[theme]])
            title = f"Words Appearing in Codes"
        agg_cloud = viz.make_cloud(text, title)
        return agg_cloud

    # user selects to make a word cloud from themes when they have aggregated theme runs
    elif source == "Themes" and run_type_state == "Themes":
        text = " ".join([theme for theme in aggregated_dict_state.keys()])
        title = f"Words Appearing in Themes"
        agg_cloud = viz.make_cloud(text, title)
        return agg_cloud

    # user selects to make a word cloud from quotes when they have aggregated total runs
    elif source == "Quotes" and run_type_state == "Total":
        agg_code_dict, agg_theme_dict = make_code_and_theme_from_total(aggregated_dict_state)
        if filter_type == "None": # do word cloud for all quotes
            text = " ".join([tup[0] for code in agg_code_dict.keys() for tup in agg_code_dict[code]])
            title = f"Words Appearing in Quotes"

        elif filter_type == "Codes":
            if filter_value != "None":
                # filter quotes by code
                text = " ".join([tup[0] for tup in agg_code_dict[filter_value]])
                title = f"Words Appearing in Quotes in Code {filter_value}"
            else: # do word cloud for all quotes
                text = " ".join([tup[0] for code in agg_code_dict.keys() for tup in agg_code_dict[code]])
                title = f"Words Appearing in Quotes"
            agg_cloud = viz.make_cloud(text, title)
            return agg_cloud

        elif filter_type == "Themes":
            if filter_value != "None":
                # filter quotes by theme
                theme_code_list = [code for code in agg_theme_dict[filter_value]]
                text = " ".join([tup[0] for code in theme_code_list for tup in agg_code_dict[code]])
                title = f"Words Appearing in Quotes in Theme {filter_value}"
            else: # do word cloud for all quotes
                text = " ".join([tup[0] for code in agg_code_dict.keys() for tup in agg_code_dict[code]])
                title = f"Words Appearing in Quotes"
            agg_cloud = viz.make_cloud(text, title)
            return agg_cloud

    # user selects to make a word cloud from codes when they have aggregated total runs
    elif source == "Codes" and run_type_state == "Total":
        _, agg_theme_dict = make_code_and_theme_from_total(aggregated_dict_state)
        if filter_value != "None":
            # filter codes by theme
            text = " ".join([code for code in agg_theme_dict[filter_value]])
            title = f"Words Appearing in Codes in Theme {filter_value}"
        else: # do word cloud for all quotes
            text = " ".join([code for theme in agg_theme_dict.keys() for code in agg_theme_dict[theme]])
            title = f"Words Appearing in Codes"
        agg_cloud = viz.make_cloud(text, title)
        return agg_cloud

    # user selects to make a word cloud from themes when they have aggregated total runs
    elif source == "Themes" and run_type_state == "Total":
        _, agg_theme_dict = make_code_and_theme_from_total(aggregated_dict_state)
        text = " ".join([theme for theme in agg_theme_dict.keys()])
        title = f"Words Appearing in Themes"
        agg_cloud = viz.make_cloud(text, title)
        return agg_cloud