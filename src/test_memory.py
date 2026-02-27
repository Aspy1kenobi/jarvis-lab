from memory import parse_note, Memory

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
    assert tag is None
    assert text == "still no tag"


def test_parse_note_empty_brackets():
    tag, text = parse_note("[] no tag here")
    assert tag is None
    assert text == "no tag here"

def test_export_to_txt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    m = Memory()
    m.data = {"notes": []}
    m.add_note("[research] attention mechanisms")
    m.add_note("unrelated note")
    
    filename, count = m.export_to_txt()
    
    assert count == 2
    assert filename is not None
    
    content = (tmp_path / filename).read_text()
    assert "attention mechanisms" in content
    assert "unrelated note" in content
    assert "[research]" in content