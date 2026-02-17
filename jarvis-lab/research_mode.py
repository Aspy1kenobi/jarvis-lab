"""
Research journal commands for JARVIS.
Track experiments, ideas, and progress.
"""

from datetime import datetime
from memory import add_note, search_notes, list_notes


def cmd_experiment(args):
    """
    Log an experiment.
    Usage: experiment <name> | <hypothesis> | <result>
    """
    if not args:
        # List recent experiments
        experiments = search_notes("experiment")
        if not experiments:
            print("No experiments logged yet.")
            print("\nUsage: experiment <name> | <hypothesis> | <result>")
            print("Example: experiment debate-agents | Multi agents produce better ideas | Improved coherence 15%")
        else:
            print("\n📊 Recent Experiments:")
            for exp in experiments[-10:]:
                print(f"  • {exp['text']}")
        return
    
    # Parse experiment format
    parts = args.split("|")
    if len(parts) != 3:
        print("Format: experiment <name> | <hypothesis> | <result>")
        print("Example: experiment debate-agents | Multi agents will produce better ideas | Needs more work")
        return
    
    name, hypothesis, result = [p.strip() for p in parts]
    
    # Log it
    note = f"[experiment] {name} | H: {hypothesis} | R: {result} | Date: {datetime.now().strftime('%Y-%m-%d')}"
    add_note(note)
    print(f"✓ Logged experiment: {name}")


def cmd_idea(args):
    """Quick capture an idea"""
    if not args:
        # Show recent ideas
        ideas = search_notes("idea")
        if ideas:
            print("\n💡 Recent Ideas:")
            for idea in ideas[-10:]:
                print(f"  • {idea['text']}")
        else:
            print("Usage: idea <your idea>")
        return
    
    add_note(f"[idea] {args}")
    print("💡 Idea captured!")


def cmd_decision(args):
    """
    Log a design decision.
    Usage: decision <what> | <why>
    """
    if not args:
        # Show recent decisions
        decisions = search_notes("decision")
        if decisions:
            print("\n📋 Recent Decisions:")
            for dec in decisions[-10:]:
                print(f"  • {dec['text']}")
        else:
            print("Usage: decision <what you decided> | <why>")
            print("Example: decision Use GPT-2 for baseline | It's free and well-documented")
        return
    
    parts = args.split("|", 1)
    if len(parts) != 2:
        print("Format: decision <what> | <why>")
        return
    
    what, why = [p.strip() for p in parts]
    note = f"[decision] {what} | Reason: {why}"
    add_note(note)
    print(f"✓ Decision logged")


def cmd_progress(args):
    """Weekly progress summary"""
    print("\n" + "="*60)
    print("📈 RESEARCH PROGRESS SUMMARY")
    print("="*60 + "\n")
    
    # Count by tag
    experiments = search_notes("experiment")
    ideas = search_notes("idea")
    decisions = search_notes("decision")
    learnings = search_notes("learning")
    
    print(f"📊 Stats:")
    print(f"  Experiments run: {len(experiments)}")
    print(f"  Ideas captured: {len(ideas)}")
    print(f"  Decisions made: {len(decisions)}")
    print(f"  Learnings noted: {len(learnings)}")
    
    if experiments:
        print("\n🔬 Recent Experiments:")
        for exp in experiments[-5:]:
            print(f"  • {exp['text'][13:]}")  # Skip [experiment] tag
    
    if ideas:
        print("\n💡 Recent Ideas:")
        for idea in ideas[-5:]:
            print(f"  • {idea['text'][7:]}")  # Skip [idea] tag
    
    if decisions:
        print("\n📋 Recent Decisions:")
        for dec in decisions[-3:]:
            print(f"  • {dec['text'][11:]}")  # Skip [decision] tag
    
    print("\n" + "="*60)


def cmd_review(args):
    """Capture what you learned"""
    if not args:
        # Show recent learnings
        learnings = search_notes("learning")
        if learnings:
            print("\n📚 Recent Learnings:")
            for learn in learnings[-10:]:
                print(f"  • {learn['text']}")
        print("\nUsage: review <what you learned>")
        return
    
    add_note(f"[learning] {args}")
    print("✓ Learning captured!")


def cmd_todo(args):
    """Track research todos"""
    if not args:
        # Show todos
        todos = search_notes("todo")
        if todos:
            print("\n✅ Research Todos:")
            for i, todo in enumerate(todos, 1):
                print(f"  {i}. {todo['text'][7:]}")  # Skip [todo] tag
        else:
            print("No todos yet!")
        print("\nUsage: todo <task>")
        return
    
    add_note(f"[todo] {args}")
    print("✓ Todo added!")
