import os
import sys

def print_directory_structure(root_path, max_files_per_folder=3, prefix="", is_last=True):
    """
    Print directory structure with limited files per folder.
    
    Args:
        root_path: Path to the root directory
        max_files_per_folder: Maximum number of files to display per folder
        prefix: Prefix for tree structure (used internally for recursion)
        is_last: Whether this is the last item in current level (used internally)
    """
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' does not exist")
        return
    
    # Get the basename of the root path
    basename = os.path.basename(root_path) or root_path
    
    # Print the root directory
    if prefix == "":
        print(basename + "/")
    
    try:
        # Get all items in directory
        items = os.listdir(root_path)
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    
    # Separate directories and files
    dirs = sorted([item for item in items if os.path.isdir(os.path.join(root_path, item))])
    files = sorted([item for item in items if os.path.isfile(os.path.join(root_path, item))])
    
    # Limit files to max_files_per_folder
    displayed_files = files[:max_files_per_folder]
    remaining_files = len(files) - max_files_per_folder
    
    # Combine directories and limited files
    all_items = dirs + displayed_files
    
    for i, item in enumerate(all_items):
        is_last_item = (i == len(all_items) - 1) and (remaining_files <= 0)
        item_path = os.path.join(root_path, item)
        
        # Determine the connector
        connector = "└── " if is_last_item else "├── "
        
        if os.path.isdir(item_path):
            print(f"{prefix}{connector}{item}/")
            # Recursively print subdirectory
            extension = "    " if is_last_item else "│   "
            print_directory_structure(item_path, max_files_per_folder, prefix + extension, is_last_item)
        else:
            print(f"{prefix}{connector}{item}")
    
    # Print remaining files indicator
    if remaining_files > 0:
        connector = "└── "
        print(f"{prefix}{connector}... and {remaining_files} more file(s)")


# Usage example
if sys.argv[1:]:
    root_directory = sys.argv[1]
else:
    root_directory = "."

print_directory_structure(root_directory, max_files_per_folder=6)