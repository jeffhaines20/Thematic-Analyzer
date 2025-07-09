import spacy
import html
import subprocess

nlp = spacy.load("en_core_web_md")

def extract_named_entities(text, selected_labels):
    doc, ner_html_output = highlight_entities(text, selected_labels)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in selected_labels]
    return entities, ner_html_output


def load_spacy_model(preferred="en_core_web_md", fallback="en_core_web_sm"):
    try:
        return spacy.load(preferred)
    except OSError:
        try:
            print(f"Downloading SpaCy model: {preferred}")
            subprocess.run(["python", "-m", "spacy", "download", preferred], check=True)
            return spacy.load(preferred)
        except Exception as e:
            print(f"Failed to load {preferred}, falling back to {fallback}. Error: {e}")
            subprocess.run(["python", "-m", "spacy", "download", fallback], check=True)
            return spacy.load(fallback)


def highlight_entities(text, selected_labels):
    ENTITY_COLORS = {
    "PERSON": "#fbb4ae",
    "ORG": "#b3cde3",
    "GPE": "#ccebc5",
    "LOC": "#decbe4",
    "DATE": "#fed9a6",
    "PRODUCT": "#ffffcc",
    "EVENT": "#e5d8bd",
    "DEFAULT": "#d9d9d9"
    }
    
    doc = nlp(text)
    spans = []

    for ent in doc.ents:
        if ent.label_ in selected_labels:
            color = ENTITY_COLORS.get(ent.label_, ENTITY_COLORS["DEFAULT"])
            span = f'<span title="{ent.label_}" style="background-color:{color}; padding:2px; border-radius:4px;">{html.escape(ent.text)}</span>'
            spans.append((ent.start_char, ent.end_char, span))

    # Sort spans and build HTML string
    spans.sort()
    last = 0
    output = ""
    for start, end, span_html in spans:
        output += html.escape(text[last:start]) + span_html
        last = end
    output += html.escape(text[last:])
    
    return doc, f"<div style='line-height:1.6; font-family:Arial, sans-serif;'>{output}</div>"

