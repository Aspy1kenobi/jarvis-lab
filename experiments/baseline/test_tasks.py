"""
Test suite for evaluating code generation quality.
10 programming tasks with test cases.
"""

TASKS = [
    {
        "id": 1,
        "name": "is_palindrome",
        "description": "Check if a string is a palindrome",
        "prompt": "def is_palindrome(s):\n    \"\"\"\n    Returns True if string s is a palindrome, False otherwise.\n    Ignores spaces and capitalization.\n    \"\"\"\n",
        "test_cases": [
            ("racecar", True),
            ("hello", False),
            ("A man a plan a canal Panama", True),
            ("", True),
            ("a", True),
            ("ab", False),
        ],
        "edge_cases": ["empty string", "single char", "spaces", "capitals"]
    },

    {
        "id": 2,
        "name": "fibonacci",
        "description": "Calculate nth Fibonacci number",
        "prompt": "def fibonacci(n):\n    \"\"\"\n    Returns the nth Fibonacci number.\n    fibonacci(0) = 0, fibonacci(1) = 1\n    \"\"\"\n",
        "test_cases": [
            (0, 0),
            (1, 1),
            (2, 1),
            (5, 5),
            (10, 55),
        ],
        "edge_cases": ["n=0", "n=1", "large n"]
    },

    {
       "id": 3,
        "name": "reverse_words",
        "description": "Reverse words in a string",
        "prompt": "def reverse_words(s):\n    \"\"\"\n    Reverses the order of words in string s.\n    Example: 'hello world' -> 'world hello'\n    \"\"\"\n",
        "test_cases": [
            ("hello world", "world hello"),
            ("a b c", "c b a"),
            ("single", "single"),
            ("", ""),
            ("  spaces  ", "spaces"),
        ],
        "edge_cases": ["empty", "single word", "extra spaces"]
    },
    
    {
        "id": 4,
        "name": "find_duplicates",
        "description": "Find duplicate elements in a list",
        "prompt": "def find_duplicates(lst):\n    \"\"\"\n    Returns a list of duplicate elements in lst.\n    Each duplicate should appear only once in the result.\n    \"\"\"\n",
        "test_cases": [
            ([1, 2, 3, 2, 4, 3], [2, 3]),
            ([1, 2, 3, 4], []),
            ([1, 1, 1], [1]),
            ([], []),
        ],
        "edge_cases": ["no duplicates", "all duplicates", "empty list"]
    },
    
    {
        "id": 5,
        "name": "merge_sorted",
        "description": "Merge two sorted lists",
        "prompt": "def merge_sorted(list1, list2):\n    \"\"\"\n    Merges two sorted lists into one sorted list.\n    \"\"\"\n",
        "test_cases": [
            ([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]),
            ([1, 2, 3], [], [1, 2, 3]),
            ([], [1, 2], [1, 2]),
            ([1], [2], [1, 2]),
        ],
        "edge_cases": ["empty lists", "single element", "different lengths"]
    },
    
    {
        "id": 6,
        "name": "is_prime",
        "description": "Check if number is prime",
        "prompt": "def is_prime(n):\n    \"\"\"\n    Returns True if n is a prime number, False otherwise.\n    \"\"\"\n",
        "test_cases": [
            (2, True),
            (3, True),
            (4, False),
            (17, True),
            (1, False),
            (0, False),
            (-5, False),
        ],
        "edge_cases": ["n=0", "n=1", "negative numbers", "n=2"]
    },
    
    {
        "id": 7,
        "name": "count_vowels",
        "description": "Count vowels in a string",
        "prompt": "def count_vowels(s):\n    \"\"\"\n    Returns the number of vowels (a, e, i, o, u) in string s.\n    Case insensitive.\n    \"\"\"\n",
        "test_cases": [
            ("hello", 2),
            ("AEIOU", 5),
            ("xyz", 0),
            ("", 0),
            ("Programming", 3),
        ],
        "edge_cases": ["no vowels", "all vowels", "mixed case", "empty"]
    },
    
    {
        "id": 8,
        "name": "binary_search",
        "description": "Binary search in sorted list",
        "prompt": "def binary_search(lst, target):\n    \"\"\"\n    Returns the index of target in sorted list lst.\n    Returns -1 if not found.\n    \"\"\"\n",
        "test_cases": [
            ([1, 3, 5, 7, 9], 5, 2),
            ([1, 3, 5, 7, 9], 1, 0),
            ([1, 3, 5, 7, 9], 9, 4),
            ([1, 3, 5, 7, 9], 4, -1),
            ([], 1, -1),
        ],
        "edge_cases": ["empty list", "first element", "last element", "not found"]
    },
    
    {
        "id": 9,
        "name": "remove_duplicates",
        "description": "Remove duplicates from list preserving order",
        "prompt": "def remove_duplicates(lst):\n    \"\"\"\n    Returns a new list with duplicates removed, preserving order.\n    \"\"\"\n",
        "test_cases": [
            ([1, 2, 2, 3, 3, 3, 4], [1, 2, 3, 4]),
            ([1, 1, 1], [1]),
            ([], []),
            ([1, 2, 3], [1, 2, 3]),
        ],
        "edge_cases": ["all same", "no duplicates", "empty"]
    },
    
    {
        "id": 10,
        "name": "factorial",
        "description": "Calculate factorial of n",
        "prompt": "def factorial(n):\n    \"\"\"\n    Returns n! (n factorial).\n    factorial(0) = 1\n    \"\"\"\n",
        "test_cases": [
            (0, 1),
            (1, 1),
            (5, 120),
            (10, 3628800),
        ],
        "edge_cases": ["n=0", "n=1", "large n"]
    },
]


def get_task(task_id):
    """Get task by ID"""
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return None


def run_test(func, test_cases):
    """
    Run test cases on a function.
    Returns (passed, total, failures)
    """
    passed = 0
    failures = []
    
    for test in test_cases:
        if len(test) == 2:
            input_val, expected = test
            args = (input_val,)
        else:
            *inputs, expected = test
            args = tuple(inputs)
        
        try:
            result = func(*args)
            if result == expected:
                passed += 1
            else:
                failures.append({
                    'input': args,
                    'expected': expected,
                    'got': result
                })
        except Exception as e:
            failures.append({
                'input': args,
                'expected': expected,
                'error': str(e)
            })
    
    return passed, len(test_cases), failures


if __name__ == "__main__":
    # Print all tasks
    print("=" * 60)
    print("PROGRAMMING TEST SUITE")
    print("=" * 60)
    print(f"\nTotal tasks: {len(TASKS)}\n")
    
    for task in TASKS:
        print(f"{task['id']}. {task['name']}")
        print(f"   {task['description']}")
        print(f"   Test cases: {len(task['test_cases'])}")
        print(f"   Edge cases: {', '.join(task['edge_cases'])}")
        print()