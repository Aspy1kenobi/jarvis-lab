import json
import os
from datetime import datetime

MEMORY_PATH = os.path.join("data", "memory.json")


def _ensure_data_dir():
    """Create data directory if it doesn't exist"""
    os.makedirs("data", exist_ok=True)


def load_memory():
    """Load memory from JSON file, return empty structure if file doesn't exist"""
    _ensure_data_dir()
    if not os.path.exists(MEMORY_PATH):
        return {"notes": []}

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory_obj):
    """Save memory object to JSON file"""
    _ensure_data_dir()
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory_obj, f, indent=2, ensure_ascii=False)


def parse_note(text):
    """
    Extract tag from note text if present.
    Format: [tag] note text
    Returns: (tag, clean_text)
    """
    text = text.strip()
    tag = None

    if text.startswith("[") and "]" in text:
        closing = text.find("]")
        tag = text[1:closing].strip()
        text = text[closing + 1:].strip()

    return tag, text


def add_note(text):
    """Add a new note to memory with optional tag"""
    memory = load_memory()
    tag, clean_text = parse_note(text)

    memory["notes"].append({
        "text": clean_text,
        "tag": tag,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    })   
    save_memory(memory)


def list_notes(limit=10):
    """Return the most recent N notes"""
    memory = load_memory()
    notes = memory.get("notes", [])
    return notes[-limit:]


def search_notes(query):
    """
    Search notes by text or tag with relevance scoring.
    Returns notes sorted by relevance (highest score first).
    Each result includes a 'score' field
    """
    memory = load_memory()
    notes = memory.get("notes", [])
    q = query.lower().strip()

    scored_results = []

    for note in notes:
        score = 0
        text_lower = note["text"].lower()

        # Score 1: Count how many times query appears in text
        text_count = text_lower.count(q)
        score += text_count * 2 # each occurrence = 2 points

        # Score 2: Bonus if query is a whole word (Not just part of a word)
        words = text_lower.split()
        if q in words:
            score += 5 # Whole word march = 5 bonus points

        # Score 3: Tag match is worth a lot
        if note.get("tag") and q in note["tag"].lower():
            score += 15 # tag match = 15 points

            # extra bonus if tag is exact match
            if q == note["tag"].lower():
                score += 10 # Exact tag match = 10 more points

        # Only include nores with score >0
        if score > 0:
            # Add the score to the note (We'll diplay it later)
            note_with_score = note.copy() #Don't modify original
            note_with_score["score"] = score
            scored_results.append(note_with_score)

    #Sort by score, highest first
    #The key=lambda tells Python: "sort by the 'score' field"
    scored_results.sort(key=lambda n: n["score"], reverse=True)

    return scored_results


def get_all_tags():
    """Return a sorted list of all unique tags"""
    memory = load_memory()
    notes = memory.get("notes", [])

    tags = set()
    for note in notes:
        tag = note.get("tag")
        if tag:
            tags.add(tag)

    return sorted(tags)


def notes_by_tag(tag):
    """Return all notes with a specific tag"""
    memory = load_memory()  # FIXED: was "load_memroy()" - typo!
    notes = memory.get("notes", [])

    tag = tag.lower().strip()
    return [note for note in notes if note.get("tag") and note["tag"].lower() == tag]


def delete_last_note():
    """Delete and return the most recent note, or None if no notes exist"""
    memory = load_memory()
    notes = memory.get("notes", [])

    if not notes:
        return None

    deleted = notes.pop()
    save_memory(memory)
    return deleted

def delete_all_notes():
    """Delete all notes and return the count deleted"""
    memory = load_memory()
    notes = memory.get("notes", [])
    
    if not notes:
        return None  # No notes to delete
    
    count = len(notes)  # How many notes there were
    memory["notes"] = []  # Clear the list
    save_memory(memory)
    return count  # Return how many we deleted

def export_to_markdown(tag=None):
    """
    Export notes to a markdown file.
    If tag is provided, only export notes with that tag.
    return: (filename, count) tuple
    """
    memory = load_memory()
    notes = memory.get("notes", [])

    # Filter by tag if provided
    if tag:
        notes = [n for n in notes if n.get("tag") and n["tag"].lower() == tag.lower()]
        filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.md"
    else:
        filename = f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.md"

    if not notes:
        return None, 0
    
    # Create markdown content
    content = f"# JARVIS Notes Export\n\n"
    content += f"**Exported:** {datetime.now.strftime('%B %d, %Y at %I:%M %p')}\n"
    content += f"**Total Notes** {len(notes)}\n\n"
    content += "---\n\n"

    #group by tag
    tagged_notes = {}
    untagged_notes = []

    for note in notes:
        tag_name = note.get("tag")
        if tag_name:
            if tag_name not in tagged_notes:
                tagged_notes[tag_name] = []
            tagged_notes[tag_name].append(note)
        else:
            untagged_notes.append(note)

    # Write untagged notes
    if untagged_notes:
        content += f"## Untagged Notes \n\n"
        for note in untagged_notes:
            timestamp = datetime.fromisoformat(note['timestamp']).strftime('%b %d, %I:%M %p')
            content += f"**{timestamp}** \n{note['text']}\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename, len(notes)

def export_to_json(tag=None):
    """ 
    Export notes to a JSON file.
    If tag is provided, only export notes with that tag.
    Returns: (filename, count) tuple
    """
    memory = load_memory()
    notes = memory.get("notes", [])

    # Filter by tag if provided
    if tag:
        notes = [n for n in notes if n.get("tag") and n["tag"].lower() == tag.lower()]
        filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.json"
    else:
        filename - f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    if not notes:
        return None, 0
    
    # Create export structure
    export_data = { 
        "exported_at": datetime.now().isoformat(),
        "note_count": len(notes),
        "filter_tag": tag,
        "notes": notes
    }

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    return filename, len(notes)

def export_to_txt(tag=None):
    """ 
    Export notes to a plain text file.
    If tag is provided, only export notes with that tag.
    Returns: (filename, count) tuple)
    """
    memory = load_memory()
    notes = memory.get("notes", [])

    # Filter by tag if provided
    if tag:
        notes = [n for n in notes if n.get("tag") and n["tag"].lower() == tag.lower()]
        filename = f"{tag}_notes_{datetime.now().strftime('%Y-%m-%d')}.txt"
    else:
        filename = f"notes_backup_{datetime.now().strftime('%Y-%m-%d')}.txt"

    if not notes:
        return None, 0
    
    # Create text context
    content = "JARVIS NOTES EXPORT\n"
    content += "=" * 60 + "\n"
    content += f"Exported: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n"
    content += f"Total Notes: {len(notes)}\n"
    content += "=" * 60 + "\n\n"

    for i, note in enumerate(notes, 1):
        timestamp = datetime.fromisoformat(note['timestamp']).strftime('%b %d, %I:%M %p')
        tag_display = f"[{note['tag']}] " if note.get('tag') else ""
        content += f"{i}, {timestamp} - {tag_display}{note['text']}\n"

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename, len(notes)
