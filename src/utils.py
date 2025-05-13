import os
import json
from typing import List, Dict, Any


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_to_file(data: Any, file_path: str) -> None:
    """Save data to a file"""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Determine the file type from extension
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        # Default to plain text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(data))


def format_fine_prints_for_output(fine_prints: List[Dict]) -> str:
    """Format fine prints for text output"""
    output = []
    
    for item in fine_prints:
        output.append(f"Source: {item['source']}, Page: {item['page']}")
        output.append("-" * 80)
        output.append(item['fine_prints'])
        output.append("\n" + "=" * 80 + "\n")
    
    return "\n".join(output)