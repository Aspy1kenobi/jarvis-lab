from colors import green, blue, red, cyan, yellow, success, error, tag, bold
from memory import add_note, list_notes, search_notes, delete_last_note, delete_all_notes, get_all_tags, notes_by_tag, load_memory, retrieve 
from agents import run_lab_meeting
from datetime import datetime
from research_mode import cmd_experiment, cmd_idea, cmd_decision, cmd_progress, cmd_review, cmd_todo
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

def print_help():
    """Display available commands"""
    print("\n📝 Core Commands:")
    print("  note [text]        - save a note to memory")
    print("  notes [tag]        - list recent notes")
    print("  search <query>     - search notes")
    print("  tags               - list all tags")
    print("  export <format>    - export notes to file")
    print("  imagine [prompt]   - generate creative text with LLM")
    print("  context on/off     - toggle context mode")
    print("  lab <topic>        - run multi-agent lab meeting")
    print("  retrieve           - run numpy based retrieval scoring")
    
    print("\n🔬 Research Mode:")
    print("  experiment         - log research experiment")
    print("  idea <text>        - quick capture idea")
    print("  decision           - log design decision")
    print("  progress           - view research summary")
    print("  review             - capture learning")
    print("  todo <task>        - track research tasks")
    
    print("\n🛠️ Utilities:")
    print("  delnote            - delete most recent note")
    print("  clear              - delete all notes")
    print("  help               - show this help")
    print("  quit               - exit\n")


# ═══════════════════════════════════════════════════════════════
# COMMAND FUNCTIONS
# Each function handles one command and receives 'args' (text after command)
# ═══════════════════════════════════════════════════════════════

def cmd_note(args):
    """
    Save a note to memory.
    If args is provided: save it directly
    If args is empty: prompt the user for text
    """
    if args:
        # User typed "note some text here"
        add_note(args)
        print(success("✓ Saved."))
    else:
        # User just typed "note"
        text = input("Enter note text: ").strip()
        if text:
            add_note(text)
            print("✓ Saved.")
        else:
            print(error("✗ Cancelled (empty note)."))


def cmd_notes(args):
    """
    List recent notes.
    If args provided: filter by that tag
    Otherwise: show last 10 notes
    """
    if args:
        # Filter by tag
        notes = notes_by_tag(args)
        if not notes:
            print(f"No notes with tag '{args}'.")
        else:
            print(f"\n{bold('Notes tagged')} {tag(f'[{args}')}:")
            _display_notes(notes)
    else:
        # Show recent notes
        notes = list_notes(limit=10)
        if not notes:
            print("No notes yet.")
        else:
            print(f"\n{bold('Recent notes')}:")
            _display_notes(notes)


def cmd_search(args):
    """Search notes by text or tag with relevance ranking"""
    if not args:
        print("Usage: search <query>")
        return
    
    results = search_notes(args)
    if not results:
        print("No matches.")
    else:
        print(f"\n{bold('Search results for')} {cyan(args)} {bold('(ranked by relevance)')}:")

        # Display with scores
        for note in results:
            timestamp = _format_timestamp(note['timestamp'])
            score = note.get('score', 0)

            # Display tag in blue if present
            if note.get('tag'):
                tag_display = tag(f"[{note['tag']}]") + " "
            else:
                tag_display = ""

            #Show Score in yellow
            print(f" . {cyan(timestamp)} - {tag_display}{note['text']} {yellow(f'score: {score}')}")


def cmd_tags(args):
    """List all unique tags"""
    tags = get_all_tags()
    if not tags:
        print("No tags yet.")
    else:
        print("\nTags:")
        for tag in tags:
            print(f"  [{tag}]")


def cmd_delnote(args):
    """Delete the most recent note"""
    deleted = delete_last_note()
    if deleted is None:
        print("No notes to delete.")
    else:
        timestamp = _format_timestamp(deleted['timestamp'])
        print(error(f"✗ Deleted: ({timestamp}) {deleted['text']}"))

def cmd_clear(args):
    """Delete all notes with confirmation"""
    # First, check if there are any notes
    memory = load_memory()  # You'll need to import load_memory
    count = len(memory.get("notes", []))
    
    if count == 0:
        print("No notes to delete.")
        return
    
    # Ask for confirmation
    confirm = input(f"⚠️  Delete all {count} notes? Type 'yes' to confirm: ").strip().lower()
    
    if confirm == "yes":
        delete_all_notes()
        print(error(f"✗ Deleted {count} note(s)."))
    else:
        print("Cancelled.")

def cmd_lab(args):
    """Run a multi-agent lab meeting on a topic"""
    if not args:
        print("Usage: lab <topic>")
        print("Example: lab build a CLI assistant")
        return
    
    print(f"\n{'='*60}")
    print(f"LAB MEETING: {args}")
    print('='*60)
    print(run_lab_meeting(args))
    print('='*60 + "\n")

def cmd_context(args):
    """Toggle Context mode or show context status"""
    # We'll need access to the context state, so we'll handle this differently
    # For now, just show usage
    if not args:
        print("Usage:")
        print("  context on      - enable context mode")
        print("  context off     - disable context mode")
        print("  context show    - display current context")
        print("  context clear   - clear context buffer")
        return
    
    # This will be filled in once we set up the context system
    print("Context system coming soon...")

def cmd_export(args):
    """Export notes to a file"""
    if not args:
        print("Usage:")
        print("  export markdown [tag]  - export to .md file")
        print("  export json [tag]      - export to .json file")
        print("  export txt [tag]       - export to .txt file")
        print("\nExamples:")
        print("  export markdown        - export all notes")
        print("  export json project    - export only [project] notes")
        return
    parts = args.split(maxsplit=1)
    format_type = parts[0].lower()
    tag = parts[1] if len(parts) > 1 else None

    # Import export functions
    from memory import export_to_markdown, export_to_json, export_to_txt

    if format_type == "markdown" or format_type == "md":
        filename, count = export_to_markdown(tag)
    elif format_type == "json":
        filename, count = export_to_json(tag)
    elif format_type == "txt" or format_type == "text":
        filename, count = export_to_txt(tag)
    else:
        print(f"Unknown format: {format_type}")
        print("Supported formats: markdown, json, txt")
        return
    
    if filename:
        tag_msg = f" with tag [{tag}]" if tag else ""
        print(success(f"✓ Exported {count} notes(s){tag_msg} to {cyan(filename)}"))
    else:
        tag_msg = f" with tag [{tag}]" if tag else ""
        print(f"No notes{tag_msg} to export.")


def cmd_imagine(args):
    """Generate creative text using the LLM"""
    from llm_bridge import generate_text, is_available, initialize_llm
    
    if not is_available():
        print(error("✗ LLM not available."))
        print("Make sure the transformer model exists at:")
        print(f"  Check that your model file exists at the expected path.")
        return
      
    # Default values
    prompt = "Once upon a time"
    length = 300
    temperature = 0.7
    
    # Parse arguments
    if args:
        parts = args.split(maxsplit=2)
        if len(parts) >= 1:
            prompt = parts[0]
        if len(parts) >= 2:
            try:
                length = int(parts[1])
            except ValueError:
                print("Invalid length, using default 300")
        if len(parts) >= 3:
            try:
                temperature = float(parts[2])
            except ValueError:
                print("Invalid temperature, using default 0.7")
    else:
        # Use recent notes as inspiration
        notes = list_notes(limit=3)
        if notes:
            prompt = notes[-1]['text'][:30]
    
    # Initialize
    initialize_llm()
    
    print(f"\n{cyan('Generating from prompt:')} {prompt}")
    print(f"{cyan('Settings:')} length={length}, temperature={temperature}")
    print(f"\n{bold('Generated Text:')}")
    print("=" * 60)
    
    text = generate_text(prompt, length, temperature)
    
    if text:
        print(text)
        print("=" * 60)
        save = input(f"\n{cyan('Save to notes? (y/n):')} ").strip().lower()
        if save == 'y':
            add_note(f"[generated] {text[:200]}")
            print(success("✓ Saved to notes"))
    else:
        print(error("✗ Failed to generate text"))


def cmd_help(args):
    """Show help text"""
    print_help()


def cmd_quit(args):
    """Exit the program"""
    print("Goodbye.")
    exit()


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _format_timestamp(timestamp_str):
    """
    Convert ISO timestamp to human-readable format.
    Example: "2025-02-06T14:30:00" → "Feb 06, 2:30 PM"
    """
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%b %d, %I:%M %p")
    except:
        # If parsing fails, return original
        return timestamp_str


def _display_notes(notes):
    """
    Display a list of notes with nice formatting and colors.
    Shows: timestamp, tag (if present), text
    """
    for note in notes:
        timestamp = _format_timestamp(note['timestamp'])
        
        if note.get('tag'):
            tag_display = tag(f"[{note['tag']}]") + " "
        else:
            tag_display = ""

        print(f" . {cyan(timestamp)} - {tag_display}{note['text']}")


# ═══════════════════════════════════════════════════════════════
# MAIN PROGRAM
# ═══════════════════════════════════════════════════════════════

def main():
    print("JARVIS-Lab (Offline) — v0.2")
    print_help()

    # Context system state
    context_enabled = False
    context_notes = []
    
    # Command registry: maps command names to their functions
    COMMANDS = {
        "note": cmd_note,
        "notes": cmd_notes,
        "search": cmd_search,
        "tags": cmd_tags,
        "delnote": cmd_delnote,
        "clear": cmd_clear,
        "context": None,
        "export": cmd_export,
        "imagine": cmd_imagine,
        "lab": None,
        "retrieve": retrieve,
        "experiment": cmd_experiment,
        "idea": cmd_idea,
        "decision": cmd_decision,
        "progress": cmd_progress,
        "review": cmd_review,
        "todo": cmd_todo,
        "help": cmd_help,
        "quit": cmd_quit,
    }

    # Main loop
    while True:
    # Show context indicator in prompt if enabled
        prompt = ">> " if not context_enabled else f"{cyan('[CTX]')} >> "
        user = input(prompt).strip()  # <- Use the prompt variable!

        # Skip empty input
        if not user:
            continue

        # Parse command and arguments
        parts = user.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Special handling for context command
        if command == "context":
            if args == "on":
                context_enabled = True
                context_notes = list_notes(limit=10)
                print(success(f"✓ Context mode enabled ({len(context_notes)} notes loaded)"))
                continue
            elif args == "off":
                context_enabled = False
                context_notes = []
                print("Context mode disabled")
                continue
            elif args == "show":
                if not context_enabled:
                    print("Context mode is OFF")
                else:
                    print(f"\n{bold('Context Buffer')} ({len(context_notes)} notes):")
                    _display_notes(context_notes)
                continue
            elif args == "clear":
                context_notes = []
                print("Context buffer cleared")
                if context_enabled:
                    context_notes = list_notes(limit=10)
                    print(f"Reloaded {len(context_notes)} notes from memory")
                continue
            else:
                cmd_context(args)
                continue

        # Special handling for lab command (pass context)
        if command == "lab":
            if not args:
                print("Usage: lab <topic>")
                print("Example: lab build a CLI assistant")
                continue
            
            # Format context for the agents
            context_str = ""
            if context_enabled and context_notes:
                context_str = "\n\nRECENT CONTEXT:\n"
                for note in context_notes[-5:]:  # Note: added : after -5
                    tag_display = f"[{note['tag']}] " if note.get('tag') else ""
                    context_str += f"- {tag_display}{note['text']}\n"
            
            print(f"\n{'='*60}")
            print(f"LAB MEETING: {args}")
            if context_enabled:
                print(cyan("(Context mode: ON)"))
            print('='*60)
            print(run_lab_meeting(args, context_str))
            print('='*60 + "\n")
            continue

        # Execute regular commands
        if command in COMMANDS:
            handler = COMMANDS[command]
            if handler:  # Skip None entries (context and lab handled above)
                handler(args)
        else:
            print(f"Unknown command: '{command}'. Type 'help' for available commands.")
if __name__ == "__main__":
    main()