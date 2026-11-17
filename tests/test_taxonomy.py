"""The engine loads its categories and prompts from the taxonomy file."""
from fmedia import taxonomy


def test_example_taxonomy_loads():
    assert taxonomy.NEGATIVE_TYPE == "not_relevant"
    assert taxonomy.POSITIVE_TYPES  # at least one positive category
    assert taxonomy.NEGATIVE_TYPE not in taxonomy.POSITIVE_TYPES
    assert taxonomy.NON_FATAL_TYPES <= taxonomy.POSITIVE_TYPES


def test_prompts_are_loaded_and_marked_as_examples():
    assert "EXAMPLE PROMPT" in taxonomy.CLASSIFIER_PROMPT
    assert "EXAMPLE PROMPT" in taxonomy.TRIAGE_PROMPT


def test_secondary_axes_have_defaults():
    assert taxonomy.RELACION_DEFAULT in taxonomy.RELACIONES
    assert taxonomy.MOTIVO_DEFAULT in taxonomy.MOTIVOS


def test_keyword_gate_is_literal_and_case_insensitive():
    from fmedia import match_patterns
    assert match_patterns.matches("An Example Keyword One appears", ["example keyword one"])
    assert not match_patterns.matches("nothing relevant here", ["example keyword one"])
