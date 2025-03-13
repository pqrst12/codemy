import xml.etree.ElementTree as ET
import json
import os
import glob
import re

def xml_to_json(xml_file_path, json_file_path=None):
    """
    Convert an XML file to clean JSON format, preserving namespaces at root and child levels.
    
    Args:
        xml_file_path (str): Path to the XML file
        json_file_path (str, optional): Path to save the JSON file. If None, will use the same name as XML but with .json extension
    
    Returns:
        str: Path to the created JSON file
    """
    if json_file_path is None:
        # Use the same filename but with .json extension
        json_file_path = os.path.splitext(xml_file_path)[0] + '.json'
    
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Convert XML to JSON with namespace preservation at root and child levels
    json_data = element_to_clean_dict(root, preserve_namespace=True)
    
    # Write JSON to file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {xml_file_path} to {json_file_path}")
    return json_file_path

def element_to_clean_dict(element, preserve_namespace=False, is_child=False):
    """
    Convert an XML element to a clean dictionary representation.
    
    Args:
        element: An ElementTree element
        preserve_namespace: Whether to preserve namespace (only at root and child levels)
        is_child: Whether the element is a direct child of the root
        
    Returns:
        dict: Clean dictionary representation of the element
    """
    result = {}
    
    # Add namespace information at root or child level
    if preserve_namespace and (not is_child):
        # Extract namespace from tag if present
        tag = element.tag
        namespace = None
        if '}' in tag:
            namespace, tag_name = tag.split('}', 1)
            namespace = namespace[1:]  # Remove the leading '{'
            result["namespace"] = namespace
    
    # Add element attributes directly to the result
    if element.attrib:
        for key, value in element.attrib.items():
            # Preserve namespace in attribute keys at root or child level
            if preserve_namespace and (not is_child) and '}' in key:
                ns, attr_name = key.split('}', 1)
                clean_key = f"ns:{format_key(attr_name)}"
            else:
                # Format attribute keys to be camelCase
                clean_key = format_key(key)
            result[clean_key] = format_value(value)
    
    # Process child elements
    child_elements = list(element)
    
    # If there are no child elements, just return the text content or empty dict
    if not child_elements:
        if element.text and element.text.strip():
            # If this is a leaf node with just text, return the text directly
            return format_value(element.text.strip())
        else:
            return result
    
    # Process child elements
    for child in child_elements:
        # Get tag name with namespace info for direct children of root
        tag = child.tag
        clean_tag = ""
        
        if preserve_namespace and (not is_child):
            if '}' in tag:
                ns, tag_name = tag.split('}', 1)
                ns = ns[1:]  # Remove the leading '{'
                clean_tag = format_key(tag_name)
                child_dict = element_to_clean_dict(child, False, True)
                # Add namespace info to child
                child_dict = {"namespace": ns, "value": child_dict} if isinstance(child_dict, dict) else {"namespace": ns, "value": child_dict}
            else:
                clean_tag = format_key(tag)
                child_dict = element_to_clean_dict(child, False, True)
        else:
            # For deeper levels, don't preserve namespace
            if '}' in tag:
                tag = tag.split('}', 1)[1]
            clean_tag = format_key(tag)
            child_dict = element_to_clean_dict(child, False, True)
        
        # Handle child elements with the same tag
        if clean_tag in result:
            if isinstance(result[clean_tag], list):
                result[clean_tag].append(child_dict)
            else:
                result[clean_tag] = [result[clean_tag], child_dict]
        else:
            result[clean_tag] = child_dict
    
    # If there's text content mixed with child elements, add it as a special key
    if element.text and element.text.strip():
        result["content"] = format_value(element.text.strip())
        
    return result

def format_key(key):
    """
    Format a key to be camelCase.
    
    Args:
        key (str): The key to format
        
    Returns:
        str: Formatted key
    """
    # Remove special characters and replace with spaces
    key = re.sub(r'[^a-zA-Z0-9]', ' ', key)
    
    # Split by spaces and capitalize each word except the first
    parts = key.split()
    if not parts:
        return ""
    
    # First part is lowercase
    formatted_key = parts[0].lower()
    
    # Rest of the parts are capitalized
    for part in parts[1:]:
        if part:
            formatted_key += part[0].upper() + part[1:].lower()
    
    return formatted_key

def format_value(value):
    """
    Format a value to the appropriate type.
    
    Args:
        value (str): The value to format
        
    Returns:
        The formatted value (could be int, float, bool, or str)
    """
    # Check if value is a number
    if value.isdigit():
        return int(value)
    
    # Check if value is a float
    try:
        float_val = float(value)
        return float_val
    except ValueError:
        pass
    
    # Check if value is a boolean
    if value.lower() in ('true', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'no', 'n', '0'):
        return False
    
    # Otherwise, return as string
    return value

def convert_all_xml_files(input_dir, output_dir=None):
    """
    Convert all XML files in a directory to clean JSON format.
    
    Args:
        input_dir (str): Directory containing XML files
        output_dir (str, optional): Directory to save JSON files. If None, will use the same directory as input
    
    Returns:
        list: Paths to all created JSON files
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files in the input directory
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    
    # Convert each XML file to JSON
    json_files = []
    for xml_file in xml_files:
        base_name = os.path.basename(xml_file)
        json_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.json')
        json_files.append(xml_to_json(xml_file, json_file))
    
    return json_files

# Example usage
if __name__ == "__main__":
    # Convert a single XML file to JSON
    xml_file = "split_text_files/split_part_1.xml"
    xml_to_json(xml_file)
    
    # Convert all XML files in a directory to JSON
    input_directory = "split_text_files"
    output_directory = "json_output"
    convert_all_xml_files(input_directory, output_directory)
