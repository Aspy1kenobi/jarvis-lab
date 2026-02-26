from memory import parse_note

def test_parse_note_with_tag():
    tag, text = parse_note("[research] attention is all you need")
    assert tag == "research"
    assert text == "attention is all you need"


def test_parse_note_without_tag():
    tag, text = parse_note("just a plain note")
    assert tag is None
    assert text == "just a plain note"


def test_parse_note_with_whitespace():
    tag, text = parse_note("  [idea]   sparse attention  ")
    assert tag == "idea"
    assert text == "sparse attention"

def test_parse_note_with_more_whitespace():
    tag, text = parse_note("[ ] still no tag")
    assert tag == None
    assert text == "still no tag"


def test_parse_note_empty_brackets():
    tag, text = parse_note("[] no tag here")
    # Based on the current implementation:
    # text[1:closing] will be "", so tag becomes ""
    assert tag == None
    assert text == "no tag here"