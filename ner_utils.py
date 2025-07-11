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

    # Sort spans and build HTML string with spacing preserved
    spans.sort()
    last = 0
    output_parts = []

    for start, end, span_html in spans:
        # Escape text between last span and current span
        raw_chunk = html.escape(text[last:start])
        raw_chunk = raw_chunk.replace("\n", "<br>")
        output_parts.append(raw_chunk)

        # Add highlighted entity span
        output_parts.append(span_html)
        last = end

    # Add remaining text after the last entity
    raw_chunk = html.escape(text[last:])
    raw_chunk = raw_chunk.replace("\n", "<br>")
    output_parts.append(raw_chunk)

    # Wrap in <div>
    output_html = "<div style='line-height:1.6; font-family:Arial, sans-serif; white-space: normal;'>" + "".join(output_parts) + "</div>"

    return doc, output_html

