
import re
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
from .relations import RELATION_SYNONYMS
from .store import KGStore
from .entities import ENTITY_TYPES, DEFAULT_TYPE

nlp = spacy.load("en_core_web_sm")

# Regex patterns for alias statements
ALIAS_PATTERNS = [
    re.compile(r"(?:known as|also known as) (\\w+)", re.IGNORECASE),
    re.compile(r"(?:nickname is|nicknamed) (\\w+)", re.IGNORECASE),
    re.compile(r"call (?:him|her|them)? (\\w+)", re.IGNORECASE),
    re.compile(r"goes by (\\w+)", re.IGNORECASE),
]

NER_TO_KG = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "LOC": "Place",
    "EVENT": "Event",
    "DATE": "Time",
    "TIME": "Time",
    "PRODUCT": "Object",
    "WORK_OF_ART": "Object",
    "LAW": "Concept",
    "LANGUAGE": "Concept",
}

def infer_type(ent_label: str) -> str:
    return NER_TO_KG.get(ent_label.upper(), DEFAULT_TYPE)

def process_text(store: KGStore, text: str):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        inferred_type = infer_type(ent.label_)
        ent_id = store.upsert_entity(ent.text, inferred_type)
        entities.append((ent.text, inferred_type, ent_id))

    # Detect relations (existing code)
    for word, canonical in RELATION_SYNONYMS.items():
        if word in text.lower() and len(entities) >= 2:
            src = entities[0][2]
            tgt = entities[1][2]
            store.add_relation(src, tgt, canonical)

    # Detect aliases
    if len(entities) >= 1:
        main_entity = entities[0]  # take the first recognized entity
        for pattern in ALIAS_PATTERNS:
            m = pattern.search(text)
            if m:
                alias = m.group(1)
                store.add_alias(main_entity[2], alias)
