from memory import parse_note, Memory, Note

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

def test_delete_last_note(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    m = Memory()
    m.data = {"notes": []}
    m.add_note("first note")
    m.add_note("second note")

    deleted = m.delete_last_note()

    # 1. the deleted note's text is correct
    assert deleted["text"] == "second note"
    
    # 2. only one note remains in memory
    assert len(m.data["notes"]) == 1
    
    # 3. calling delete_last_note on an empty Memory returns None
    m.delete_last_note()  # removes the remaining "first note"
    assert m.delete_last_note() is None

def test_note_dataclass_roundtrip():
    note = Note(text="attention mechanisms", tag="research", timestamp="2026-02-28T10:00:00")

    d = note.to_dict()
    assert d == {"text": "attention mechanisms", "tag": "research", "timestamp": "2026-02-28T10:00:00"}

    restored = Note.from_dict(d)
    assert restored.text == "attention mechanisms"
    assert restored.tag == "research"
    assert restored == note