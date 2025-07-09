import pandas as pd
import networkx as nx
from pyvis.network import Network
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import re
import base64
from IPython.display import display
import plotly.graph_objects as go
import plotly.colors as pc
from docx.shared import RGBColor
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import hashlib
from collections import defaultdict
from typing import Dict, List
import html
from html import escape
import run_management as rm
import matplotlib.cm as cm
import gradio as gr


def experimental_plots():
    df = pd.read_csv("assets/Experiment Results/Experiment Results.csv", index_col=0)
    plotly_plot = px.scatter(
          df,
          x="Time (Seconds)",
          y="Percentage of Text Covered",
          color="Codes Generated",
          hover_data=["New Tokens", "Chunk Size", "Time (Seconds)", "Percentage of Text Covered", "Codes Generated"],
          title="Time vs. % Covered",
          color_continuous_scale='Viridis_r',  # <- the "_r" reverses the color scale
    )

    plotly_plot.update_traces(marker=dict(size=10))

    pivot_table = df.pivot_table(
        index="New Tokens", 
        columns="Chunk Size", 
        values="Percentage of Text Covered", 
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis_r", ax=ax)
    ax.set_title("Heatmap of Text Coverage")
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("New Tokens")
    ax.invert_yaxis()

    return plotly_plot, fig

def dict_to_table(combined_dict):
    rows = []
    for theme in combined_dict.keys():
        summary = combined_dict[theme][0]['summary']
        for code in combined_dict[theme][1].keys():
            for tup in combined_dict[theme][1][code]:
                rows.append({'Theme': theme, 'Theme Summary': summary, 'Code': code, 'Quote': tup[0], 'Confidence': tup[1]})

    theme_df = pd.DataFrame(rows)
    return theme_df


def generate_theme_color(theme: str) -> str:
    """Generate a consistent hex color from a theme name."""
    hash_object = hashlib.md5(theme.encode())
    return f'#{hash_object.hexdigest()[:6]}'  # Use first 6 hex digits


def blend_colors(hex_colors):
    # Convert hex to RGB and average
    rgb_colors = [
        tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        for c in hex_colors
    ]
    n = len(rgb_colors)
    avg_rgb = tuple(sum(c[i] for c in rgb_colors) // n for i in range(3))
    return '#{:02x}{:02x}{:02x}'.format(*avg_rgb)


def highlight_text_by_theme(
    full_text: str,
    code_dict: Dict[str, List[str]],
    theme_dict: Dict[str, List[str]]
) -> str:
    # Map each code to its theme and theme color
    code_to_theme = {}
    theme_colors = {}

    for theme, codes in theme_dict.items():
        color = generate_theme_color(theme)
        theme_colors[theme] = color
        for code in codes:
            if code.lower() not in code_to_theme:
                code_to_theme[code.lower()] = []
            code_to_theme[code.lower()].append((theme, color))

    # Collect all code-tagged text (quotes) and map to theme
    sentence_theme_map = defaultdict(list)

    for code, tups in code_dict.items():
        code_lower = code.lower()
        if code_lower in code_to_theme:
            themes = [x[0] for x in code_to_theme[code_lower]]
            colors = [x[1] for x in code_to_theme[code_lower]]
            for tup in tups:
                for i, theme in enumerate(themes):
                    sentence_theme_map[tup[0]].append((theme, colors[i]))

    # Sort by sentence length to avoid nested highlights
    sorted_sentences = sorted(sentence_theme_map.keys(), key=len, reverse=True)

    highlighted_text = full_text
    for sent in sorted_sentences:
        themes_colors = sentence_theme_map[sent]
        if not themes_colors:
            continue

        # Use first associated theme color (could be extended to support multi-tag display)
        themes = [x[0] for x in themes_colors]
        colors = [x[1] for x in themes_colors]
        escaped_sent = re.escape(sent.strip())
        # Build tooltip string
        if len(themes) == 1:
            tooltip = "Theme: " + themes[0]
        else:
            tooltip = "Themes: " + " | ".join(themes)

        # Determine background color
        if len(colors) == 1:
            blended_color = colors[0]
        else:
            blended_color = blend_colors(colors)

        span = (
            f'<span style="background-color: {blended_color}; padding:2px; border-radius:4px;" '
            f'title="{tooltip}">{sent.strip()}</span>'
        )
        # Replace only once to avoid multiple highlights
        highlighted_text = re.sub(escaped_sent, span, highlighted_text, count=1)

    return f"<div style='white-space: pre-wrap; font-family: Arial, sans-serif;'>{highlighted_text}</div>"


def visualize_theme_network(combined_dict):
    G = nx.Graph()

    # Add edges: theme ↔ tag
    for theme in combined_dict.keys():
        for code in combined_dict[theme][1].keys():
            G.add_edge(theme, code)

    # Node styling
    theme_nodes = list(combined_dict.keys())
    code_nodes = [code for theme in theme_nodes for code in combined_dict[theme][1].keys()]

    pos = nx.kamada_kawai_layout(G)  # layout for more space
    #pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, nodelist=theme_nodes, node_color='skyblue', node_size=1200, label='Themes')
    nx.draw_networkx_nodes(G, pos, nodelist=code_nodes, node_color='lightgreen', node_size=600, label='Codes')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6)

    plt.title("Code and Theme Clusters", fontsize=14)
    plt.axis("off")
    plt.legend()
    plt.show()


def render_dataframe_as_html(df):
    html = df.to_html(
        escape=False,  # Allows HTML tags in the content
        index=False,   # Hide the index column
        classes="dataframe"  # Optional, useful for styling
    )

    # Wrap in a scrollable div with padding and borders
    styled_html = f"""
    <div style="max-height: 600px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
        {html}
    </div>
    """
    return styled_html


def visualize_theme_network_for_gradio(code_dict, theme_dict, filter_themes=None) -> str:
    net = Network(
        height="700px",
        width="100%",
        notebook=False,
        directed=False,
        cdn_resources="in_line"
    )

    themes_to_display = filter_themes if filter_themes else theme_dict.keys()

    for theme in themes_to_display:
        net.add_node(theme, label=theme, color='skyblue', shape='box', size=25)

        for code in theme_dict.get(theme, []):
            try:
                code_data = code_dict[code]
                if isinstance(code_data[0], tuple):
                    confidence = np.mean([x[1] for x in code_data])
                    tooltip = f"Confidence: {confidence:.2f}"
                else:
                    tooltip = "No confidence score"

                net.add_node(code, label=code, color='lightgreen', shape='ellipse', size=15, title=tooltip)
                net.add_edge(theme, code)
            except (KeyError, IndexError, TypeError):
                continue

    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 18}
      },
      "edges": {
        "color": {"inherit": true},
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "stabilization": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 100
        }
      }
    }
    """)

    html_content = net.generate_html()
    encoded = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    iframe_html = f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="700px" style="border:none;"></iframe>'

    return iframe_html


def generate_highlighted_html(full_text, code_dict):
    # Prepare highlights
    highlights = []
    for code, entries in code_dict.items():
        for sentence, conf in entries:
            highlights.append((sentence, code, conf))

    highlights.sort(key=lambda x: len(x[0]), reverse=True)
    highlighted_text = escape(full_text)

    # Replace sentences with <span> that includes tooltip
    for sentence, code, conf in highlights:
        escaped_sentence = escape(sentence)
        clean_code = re.sub(r"</?code>", "", code)
        clean_code = re.sub(r"_", " ", clean_code)
        clean_code = clean_code.title()
        tooltip = f"{clean_code} (conf: {conf:.2f})"
        span = (
            f'<span class="highlight" data-tooltip="{tooltip}">{escaped_sentence}</span>'
        )
        highlighted_text = highlighted_text.replace(escaped_sentence, span, 1)

    # Wrap in HTML with embedded CSS for tooltip
    return f"""
    <style>
    .highlight {{
        background-color: yellow;
        position: relative;
        cursor: help;
    }}
    .highlight::after {{
        content: attr(data-tooltip);
        position: absolute;
        background: #333;
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        white-space: nowrap;
        z-index: 10;
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
        top: 100%;
        left: 0;
    }}
    .highlight:hover::after {{
        opacity: 1;
    }}
    </style>
    <div style='white-space: pre-wrap; font-family: sans-serif;'>{highlighted_text}</div>
    """


def visualize_theme_network_interactive(
    combined_dict: list[dict],
    output_file: str="themes_network.html",
    filter_themes=None
):
    net = Network(
        height="700px",
        width="100%",
        notebook=True,
        directed=False,
        cdn_resources="in_line"
    )

    # Optional filtering
    themes_to_display = filter_themes if filter_themes else combined_dict.keys()

    for theme in combined_dict.keys():
        if theme not in themes_to_display:
            continue

        net.add_node(theme, label=theme, color='skyblue', shape='box', size=25)

        for code in combined_dict[theme][1].keys():
            # Support either (tag, confidence) or just tag string
            code_data = combined_dict[theme][1][code]
            if isinstance(code_data[0], tuple):
                confidence = np.mean([x[1] for x in code_data])
                tooltip = f"Confidence: {confidence}"
            else:
                tooltip = "No confidence score"

            net.add_node(code, label=code, color='lightgreen', shape='ellipse', size=15, title=tooltip)
            net.add_edge(theme, code)

    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 18}
      },
      "edges": {
        "color": {"inherit": true},
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "stabilization": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 100
        }
      }
    }
    """)

    net.show(output_file)


def visualize_sankey(combined_dict: list[dict]):
    if combined_dict is None:
        return None, gr.update(visible=True), "❌ Error: No run to build charts from. Please click 'Summarize Themes' first."
    elif len(combined_dict) == 0:
        return None, gr.update(visible=True), "❌ Error: No run to build charts from. Please click 'Summarize Themes' first."

    labels = []
    label_to_index = {}
    sources = []
    targets = []
    values = []
    node_colors = []
    link_labels = []

    # Generate distinct colors for themes
    theme_colors = pc.qualitative.Plotly  # ['#1f77b4', '#ff7f0e', ...]
    theme_to_color = {}

    theme_index = 0

    for theme in combined_dict.keys():
        if theme not in label_to_index:
            label_to_index[theme] = len(labels)
            labels.append(theme)
            color = theme_colors[theme_index % len(theme_colors)]
            theme_to_color[theme] = color
            node_colors.append(color)
            theme_index += 1

        for code in combined_dict[theme][1].keys():
            code_data = combined_dict[theme][1][code]
            if isinstance(code_data[0], tuple):
                confidence = np.mean([x[1] for x in code_data])
            else:
                confidence = 1

            if code not in label_to_index:
                label_to_index[code] = len(labels)
                labels.append(code)
                node_colors.append("#D3D3D3")  # Light gray for tags

            sources.append(label_to_index[theme])
            targets.append(label_to_index[code])
            values.append(confidence)
            link_labels.append(f"{code}: confidence {confidence}")

    # Build Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            hoverinfo="none"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color="lightgreen"
        ))])

    fig.update_layout(title_text="Thematic Clusters - Sankey Diagram", font_size=14, height=600)
    return fig, gr.update(visible=False), None


def plot_n_gram(n: int, n_gram: int, text: list):
    count_vec = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words='english').fit(text)
    bag_of_words = count_vec.transform(text)
    total = bag_of_words.sum(axis=0)
    frequencies = [(word, total[0, i]) for word, i in count_vec.vocabulary_.items()]
    frequencies =sorted(frequencies, key = lambda x: x[1], reverse=True)
    x,y=map(list,zip(*frequencies[0:n]))

    p = sns.dark_palette("#69d")
    sns.barplot(x=y,y=x, palette = p).set_title(f'{n} Most Common {n_gram}-Grams Excluding Stop Words');


def make_cloud(text: str, title: str):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig


def make_word_cloud(combined_dict, full_text, mode="text", filter_type="None", filter_value="None"):
    if combined_dict is None:
        return None, gr.update(visible=True), "❌ Error: No run to build charts from. Please click 'Summarize Themes' first."
    elif len(combined_dict) == 0:
        return None, gr.update(visible=True), "❌ Error: No run to build charts from. Please click 'Summarize Themes' first."

    if mode == "Text":
        word_cloud = make_cloud(text=full_text, title="Words in Text")

    elif mode == "Quotes":
        df = dict_to_table(combined_dict)
        if filter_type == "None":
            quotes = " ".join([text for text in df['Quote']])
            word_cloud = make_cloud(text=quotes, title="Words in Quotes in Coded Sentences")
        elif filter_type == "Codes":
            if filter_value == "None":
                quotes = " ".join([text for text in df['Quote']])
                word_cloud = make_cloud(text=quotes, title="Words in Quotes in Coded Sentences")
            else:
                # the filter_value is set to a specific code
                quotes = " ".join([quote for quote, code in zip(df["Quote"], df["Code"]) if code == filter_value])
                word_cloud = make_cloud(text=quotes, title=f"Words in Quotes in Code {filter_value}")
        elif filter_type == "Themes":
            if filter_value == "None":
                quotes = " ".join([text for text in df['Quote']])
                word_cloud = make_cloud(text=quotes, title="Words in Quotes in Coded Sentences")
            else:
                # the filter value is set to a specific theme
                quotes = " ".join([quote for quote, theme in zip(df["Quote"], df["Theme"]) if theme == filter_value])
                word_cloud = make_cloud(text=quotes, title=f"Words in Quotes in Theme {filter_value}")

    elif mode == "Codes":
        df = dict_to_table(combined_dict)
        if filter_type == "None":
            codes = " ".join([text for text in df['Code']])
            word_cloud = make_cloud(text=codes, title="Words in Codes")
        elif filter_type == "Themes":
            if filter_value == "None":
                codes = " ".join([text for text in df['Code']])
                word_cloud = make_cloud(text=codes, title="Words in Codes")
            else:
                # the filter value is set to a specific theme
                codes = " ".join([code for code, theme in zip(df["Code"], df["Theme"]) if theme == filter_value])
                word_cloud = make_cloud(text=codes, title=f"Words in Codes in Theme {filter_value}")

    elif mode == "Themes":
        df = dict_to_table(combined_dict)
        themes = " ".join([text for text in df['Theme']])
        word_cloud = make_cloud(text=themes, title="Words in Themes")
    
    else:
        return None, gr.update(visible=True), "❌ Error: Mode not recognized. Please choose a wordcloud mode."

    return word_cloud, gr.update(visible=False), None


## Aggregate visualizations
def get_theme_code_df(theme_code_freqs):
    sorted_themes = sorted(theme_code_freqs.items(), key=lambda x: x[1], reverse=True)
    theme_code_pair, freqs = zip(*sorted_themes)
    theme_code_data = []
    for i, theme_code in enumerate(theme_code_pair):
        theme_code_data.append({"Theme-Code": theme_code, "Frequency": freqs[i]})

    return pd.DataFrame(theme_code_data)


def get_code_quote_df(code_quote_freqs, max_label_length=60):
    sorted_codes = sorted(code_quote_freqs.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    code_quote_pair, freqs = zip(*sorted_codes)
    
    code_quote_data = []
    for i, code_quote in enumerate(code_quote_pair):
        # Truncate long code-quote strings
        display_text = code_quote if len(code_quote) <= max_label_length else code_quote[:max_label_length - 3] + "..."
        code_quote_data.append({
            "Code-Quote": display_text,
            "Frequency": freqs[i][0],
            "Confidence": freqs[i][1],
            "Full Text": code_quote  # Optional: keep full version for reference
        })

    return pd.DataFrame(code_quote_data)


def visualize_aggregate(run_view, aggregated_dict):
    if aggregated_dict is None:
        return None, None, None, None, None, None, "❌ Error 1: No runs to aggregate."
    
    elif len(aggregated_dict) > 0:
        run_type = str(run_view["Type"][0]).strip()

    else:
        # No runs to aggregate
        return None, None, None, None, None, None,  "❌ Error 2: No runs to aggregate."

    if run_type == "Codes":
        # structure is {code: [list of (sentence, confidence) tuples]}
        # code frequency plot
        code_freq_plot = plot_agg_freqs(aggregated_dict, agg_type = "Codes")

        # code-quote freqs with average confidence scores
        code_quote_freqs = rm.get_code_quote_freqs(aggregated_dict)
        code_quote_freq_plot = plot_agg_paired_freqs(code_quote_freqs)

        # dataframe
        code_quote_data = get_code_quote_df(code_quote_freqs)

        return code_freq_plot, code_quote_freq_plot, code_quote_data, None, None, None, "✅ Successfully visualized codes and quotes!"

    elif run_type == "Themes":
        # structure is {theme: [list of codes]}
        # theme frequency plots
        theme_freq_plot = plot_agg_freqs(aggregated_dict, agg_type = "Themes")
        theme_code_freqs = rm.get_theme_code_freqs(aggregated_dict)
        theme_code_freq_plot = plot_agg_paired_freqs(theme_code_freqs)

        # dataframe
        theme_code_data = get_theme_code_df(theme_code_freqs)

        return None, None, None, theme_freq_plot, theme_code_freq_plot, theme_code_data, "✅ Successfully visualized themes!"

    elif run_type == "Total":
        # do both of above
        # first, reverse engineer the code and theme dicts
        agg_code_dict, agg_theme_dict = rm.make_code_and_theme_from_total(aggregated_dict)

        ## now plot ##
        # codes
        code_freq_plot = plot_agg_freqs(agg_code_dict, agg_type = "Codes")

        # code-quote freqs with average confidence scores
        code_quote_freqs = rm.get_code_quote_freqs(agg_code_dict)
        code_quote_freq_plot = plot_agg_paired_freqs(code_quote_freqs)

        # dataframe
        code_quote_data = get_code_quote_df(code_quote_freqs)

        # themes
        theme_freq_plot = plot_agg_freqs(agg_theme_dict, agg_type = "Themes")
        theme_code_freqs = rm.get_theme_code_freqs(agg_theme_dict)
        theme_code_freq_plot = plot_agg_paired_freqs(theme_code_freqs)

        # dataframe
        theme_code_data = get_theme_code_df(theme_code_freqs)

        return code_freq_plot, code_quote_freq_plot, code_quote_data, theme_freq_plot, theme_code_freq_plot, theme_code_data, "✅ Successfully visualized quotes, codes, and themes!"

    return None, None, None, None, None, None, "❌ Error: Run type not recognized."


def plot_agg_freqs(agg_dict, agg_type: str):
    # this works for both code and theme aggregated dictionaries
    freqs = {key: len(values) for key, values in agg_dict.items()}
    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

    keys, freqs = zip(*sorted_freqs[:20])  # top 20
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(keys[::-1], freqs[::-1], color="skyblue")
    ax.set_title(f"Most Frequent {agg_type}")
    ax.set_xlabel("Count")
    ax.set_ylabel(f"{agg_type}")
    plt.tight_layout()

    return fig


def plot_agg_paired_freqs(paired_freqs, max_label_length=60):
    sorted_freqs = sorted(paired_freqs.items(), key=lambda x: x[1] if isinstance(x[1], int) else (x[1][0],x[1][1]), reverse=True)
    pairs, counts, conf_scores = [], [], []

    # Detect type of data
    has_conf = all(isinstance(v, list) and len(v) == 2 for v in paired_freqs.values())
    agg_type = "Code-Quote" if has_conf else "Theme-Code"

    for k, v in sorted_freqs[:20]:
        label = k if len(k) <= max_label_length else k[:max_label_length - 3] + "..."
        pairs.append(label)
        if has_conf:
            counts.append(v[0])
            conf_scores.append(v[1])
        else:
            counts.append(v)

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_conf:
        # Normalize confidence scores to [0, 1] for color mapping
        conf_norm = (np.array(conf_scores) - 1.0) / (5.0 - 1.0)
        colors = cm.Blues(conf_norm)
        ax.barh(pairs[::-1], counts[::-1], color=colors)
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=1.0, vmax=5.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Confidence Score')
    else:
        ax.barh(pairs[::-1], counts[::-1], color="skyblue")

    ax.set_title(f"Most Frequent {agg_type} Pairs")
    ax.set_xlabel("Count")
    ax.set_ylabel(f"{agg_type}")
    plt.tight_layout()
    plt.close(fig)
    return fig

