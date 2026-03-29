import os
import ast
import json
from collections import Counter

# --- CONFIGURATION ---
IGNORE_DIRS = {'data', '__pycache__', '.git', '.venv', 'venv', 'node_modules', 'outputs', 'models'}
IGNORE_FILES = {'.DS_Store', 'summarize_repo.py', 'GEMINI.md', 'LICENSE', 'README.md', 'project.toml', 'setup.py', '.gitignore'}
EXTENSIONS = {'.py', '.ipynb'}

# Storage for cross-referencing
INTERNAL_FUNCTIONS = set()
project_call_registry = Counter()

def first_pass_index_definitions(root_dir):
    """Scan the entire project to find every function/class you created."""
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                INTERNAL_FUNCTIONS.add(node.name)
                            elif isinstance(node, ast.ClassDef):
                                INTERNAL_FUNCTIONS.add(node.name)
                    except: continue

def get_ast_details(source_code, is_notebook=False):
    """Finds definitions and counts calls only if they are in our 'Internal' list."""
    try:
        tree = ast.parse(source_code)
    except:
        return [], []
    
    defined, called = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.append(f"{node.name}()")
        elif isinstance(node, ast.ClassDef):
            defined.append(f"Class:{node.name}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name): name = node.func.id
            elif isinstance(node.func, ast.Attribute): name = node.func.attr
            
            # ONLY count and report if it's one of your custom functions
            if name in INTERNAL_FUNCTIONS:
                called.append(name)
                project_call_registry[name] += 1
    
    return sorted(list(set(defined))), sorted(list(set(called)))

def get_python_structure(filepath):
    """Clean module summary: just definitions and docstrings."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        try: tree = ast.parse(content)
        except: return "  - Error parsing."

    # Update global call registry without returning the 'Logic uses' string
    get_ast_details(content)

    defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            line = f"  - Function: `{node.name}()` {' | ' + doc.splitlines()[0] if doc else ''}"
            defs.append(line)
        elif isinstance(node, ast.ClassDef):
            class_str = [f"  - Class: `{node.name}`"]
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    doc = ast.get_docstring(item)
                    class_str.append(f"    - Method: `{item.name}()` {' | ' + doc.splitlines()[0] if doc else ''}")
            defs.append("\n".join(class_str))
    
    return "\n".join(defs) if defs else "  - Contains helper logic/imports."

def get_notebook_summary(filepath):
    """Notebook layout: Markdown Hint -> Code Hint per cell."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            cells = data.get('cells', [])
        except: return "  - Error reading notebook."

    summary = []
    current_md = None
    code_idx = 1

    for cell in cells:
        if cell['cell_type'] == 'markdown':
            lines = [l.strip() for l in cell['source'] if l.strip()]
            if lines: current_md = lines[0].strip('# ').strip()[:75]
        
        elif cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if not source.strip(): continue
            
            defs, calls = get_ast_details(source)
            
            # Only include the cell if it actually does something interesting
            if current_md or defs or calls:
                cell_report = [f"  - Cell {code_idx}:"]
                if current_md: cell_report.append(f"    - Markdown Hint: {current_md}...")
                if defs: cell_report.append(f"    - Defines: {', '.join(defs)}")
                if calls: cell_report.append(f"    - Code Hint (Internal): {', '.join(calls)}")
                summary.append("\n".join(cell_report))
                code_idx += 1
            current_md = None 

    return "\n".join(summary) if summary else "  - Empty notebook."

def generate_summary(root_dir):
    # First, find all your functions so we can ignore 'max', 'plt', etc.
    first_pass_index_definitions(root_dir)
    
    output = ["# Project Summary:\n"]
    sections = []
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = sorted([d for d in dirs if d not in IGNORE_DIRS])
        rel_path = os.path.relpath(root, root_dir)
        header_path = "Root" if rel_path == "." else rel_path
        
        file_summaries = []
        for file in sorted(files):
            if file in IGNORE_FILES: continue
            file_ext = os.path.splitext(file)[1]
            if file_ext in EXTENSIONS:
                file_info = [f"### File: `{file}`"]
                if file_ext == '.py':
                    file_info.append(get_python_structure(os.path.join(root, file)))
                elif file_ext == '.ipynb':
                    file_info.append(get_notebook_summary(os.path.join(root, file)))
                file_summaries.append("\n".join(file_info))
        
        if file_summaries:
            sections.append(f"## Directory: {header_path}\n" + "\n\n".join(file_summaries))

    output.extend(sections)
    output.append("\n---\n## 📊 Internal Research Logic Frequency")
    output.append("Functions you wrote, sorted by how often they are used in this project:")
    
    for func, count in project_call_registry.most_common():
        output.append(f"* `{func}`: {count} calls")
        
    return "\n".join(output)

if __name__ == "__main__":
    report = generate_summary(os.getcwd())
    with open("Project_Summary.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ Research-Focused Summary Generated!")