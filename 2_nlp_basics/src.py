"""
WEEK 2

- Assumes you've pip installed requirements file (pip install -r requirements.txt)
- Also need to download English trained pipeline https://spacy.io/models/en#en_core_web_sm. This can be done using
    `python -m spacy download en_core_web_sm` which downloads the model as another pip package essentially.
"""
from PyPDF2 import PdfReader
import spacy
from spacy import displacy

# Assumes this model is already downloaded
nlp = spacy.load("en_core_web_sm")

def pdf_entity_extractor(pdf_file_location) -> dict[str, list[str]]:
    """
    Return a dictionary of entities from a pdf.
    """
    # Read PDF text
    reader = PdfReader(pdf_file_location)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Process with spaCy
    doc = nlp(text)

    # Collect entities
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)

    return entities

def token_analyzer(text: str) -> dict:
    """
    Given a text string, split into tokens and return for each token.
    """
    doc = nlp(text)
    tokens_info = {}

    for token in doc:
        tokens_info[token.text] = {
            "lemma": token.lemma_,
            "pos": token.pos_,
            "shape": token.shape_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop
        }

    return tokens_info

def remove_stop_words(text: str) -> str:
    """
    Remove stop words from the given text.
    """
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered_tokens)

def lemmatizer(text: str) -> str:
    """
    Lemmatize sentence.
    """
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

def create_character_tokenizer(training_text: str, text_to_tokenize: str) -> list[int]:
    """
    Create a character-level tokenizer using indices from the training text.
    """
    # Build mapping from character → index (based on first occurrence)
    char_to_index = {}
    for i, ch in enumerate(training_text):
        if ch not in char_to_index:
            char_to_index[ch] = i

    # Convert text_to_tokenize using the mapping
    tokens = [char_to_index[ch] for ch in text_to_tokenize if ch in char_to_index]

    return tokens
