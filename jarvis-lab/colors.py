"""
Simple color helpers for terminal output
"""

# Color codes - NOTE: Must be lowercase 'm' not uppercase 'M'
GREEN = "\033[32m"
BLUE = "\033[34m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Disable colors on Windows Command Prompt (they don't work well there)
import platform
if platform.system() == "Windows":
    GREEN = BLUE = RED = YELLOW = CYAN = MAGENTA = BOLD = RESET = ""


def green(text):
    """Return text in green"""
    return f"{GREEN}{text}{RESET}"

def blue(text):
    """Return text in blue"""
    return f"{BLUE}{text}{RESET}"

def red(text):
    """Return text in red"""
    return f"{RED}{text}{RESET}"

def yellow(text):
    """Return text in yellow"""
    return f"{YELLOW}{text}{RESET}"

def cyan(text):
    """Return text in cyan (light blue)"""
    return f"{CYAN}{text}{RESET}"

def bold(text):
    """Return text in bold"""
    return f"{BOLD}{text}{RESET}"


# Semantic helpers (describe what the color means, not what color it is)
def success(text):
    """Green text for success messages"""
    return green(text)

def error(text):
    """Red text for errors"""
    return red(text)

def info(text):
    """Cyan text for informational messages"""
    return cyan(text)

def tag(text):
    """Blue text for tags"""
    return blue(text)