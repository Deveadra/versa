RELATION_CONSTRAINTS = {
    "spouse_of": ("Person", "Person"),
    "has_sibling": ("Person", "Person"),
    "parent_of": ("Person", "Person"),
    "child_of": ("Person", "Person"),
    "works_at": ("Person", "Organization"),
    "employs": ("Organization", "Person"),
    "located_in": ("Place", "Place"),
    "happens_at": ("Event", "Time"),
    "owns": ("Person", "Object"),
}

RELATION_INVERSES = {
    "spouse_of": "spouse_of",  # symmetric
    "has_sibling": "has_sibling",  # symmetric
    "parent_of": "child_of",
    "child_of": "parent_of",
    "works_at": "employs",
    "employs": "works_at",
}

# Add natural language hints for queries
RELATION_QUERY_HINTS = {
    "mom": "parent_of",
    "mother": "parent_of",
    "dad": "parent_of",
    "father": "parent_of",
    "parent": "parent_of",
    "son": "child_of",
    "daughter": "child_of",
    "child": "child_of",
    "wife": "spouse_of",
    "husband": "spouse_of",
    "spouse": "spouse_of",
    "partner": "spouse_of",
    "sister": "has_sibling",
    "brother": "has_sibling",
    "sibling": "has_sibling",
    "boss": "employs",
    "manager": "employs",
    "employee": "works_at",
    "job": "works_at",
}

RELATION_SYNONYMS = {
    # Marriage
    "married_to": "spouse_of",
    "husband_of": "spouse_of",
    "wife_of": "spouse_of",
    "spouse_of": "spouse_of",
    # Family
    "sister": "has_sibling",
    "brother": "has_sibling",
    "sibling_of": "has_sibling",
    "has_sibling": "has_sibling",
    "parent_of": "parent_of",
    "father_of": "parent_of",
    "mother_of": "parent_of",
    "child_of": "child_of",
    "son_of": "child_of",
    "daughter_of": "child_of",
    # Work
    "works_at": "works_at",
    "employee_of": "works_at",
    "employs": "employs",
}
