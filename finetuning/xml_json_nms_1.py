import xml.etree.ElementTree as ET
import json
import os
import glob
import re

def xml_to_json(xml_file_path, json_file_path=None):
    """
    Convert an XML file to clean JSON format with namespace information at root level.
    
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
    
    # Extract namespaces
    namespaces = extract_namespaces(root)
    
    # Convert XML to JSON with namespaces
    json_data = {
        "namespaces": namespaces,
        **element_to_clean_dict(root, namespaces)
    }
    
    # Write JSON to file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {xml_file_path} to {json_file_path}")
    return json_file_path

def extract_namespaces(element):
    """
    Extract namespace information from XML element.
    
    Args:
        element: An ElementTree element
        
    Returns:
        dict: Dictionary of namespace prefixes and URIs
    """
    namespaces = {}
    
    # Extract namespaces from the entire tree
    for elem in element.iter():
        # Check the element's tag for namespace
        if '}' in elem.tag:
            ns_uri = elem.tag.split('}', 1)[0].strip('{')
            # Find the prefix for this namespace
            for prefix, uri in elem.nsmap.items() if hasattr(elem, 'nsmap') else []:
                if uri == ns_uri:
                    namespaces[prefix if prefix else "default"] = uri
        
        # Check attributes for namespaces
        for attr_name in elem.attrib:
            if '}' in attr_name:
                ns_uri = attr_name.split('}', 1)[0].strip('{')
                for prefix, uri in elem.nsmap.items() if hasattr(elem, 'nsmap') else []:
                    if uri == ns_uri:
                        namespaces[prefix if prefix else "default"] = uri
    
    # If ElementTree doesn't expose nsmap, extract from the root element directly
    if not namespaces and hasattr(element, 'attrib'):
        for attr_name, value in element.attrib.items():
            if attr_name.startswith('xmlns:'):
                prefix = attr_name.split(':', 1)[1]
                namespaces[prefix] = value
            elif attr_name == 'xmlns':
                namespaces["default"] = value
    
    return namespaces

def get_namespace_info(tag):
    """
    Extract namespace URI and local name from a tag.
    
    Args:
        tag (str): The tag with possible namespace
        
    Returns:
        tuple: (namespace_uri, local_name) or (None, tag) if no namespace
    """
    if '}' in tag:
        ns_uri = tag.split('}', 1)[0].strip('{')
        local_name = tag.split('}', 1)[1]
        return ns_uri, local_name
    else:
        return None, tag

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

def get_namespace_prefix(uri, namespaces):
    """
    Get namespace prefix from URI.
    
    Args:
        uri (str): Namespace URI
        namespaces (dict): Dictionary of namespace prefixes and URIs
        
    Returns:
        str: Namespace prefix or None if not found
    """
    for prefix, namespace_uri in namespaces.items():
        if namespace_uri == uri:
            return prefix
    return None

def element_to_clean_dict(element, namespaces):
    """
    Convert an XML element to a clean dictionary representation with namespace info at root level.
    
    Args:
        element: An ElementTree element
        namespaces: Dictionary of namespace prefixes and URIs
    
    Returns:
        dict: Clean dictionary representation of the element
    """
    result = {}
    
    # Get namespace info for this element
    ns_uri, local_name = get_namespace_info(element.tag)
    if ns_uri:
        ns_prefix = get_namespace_prefix(ns_uri, namespaces)
        if ns_prefix:
            result["_namespace"] = ns_prefix
    
    # Add element attributes
    attributes = {}
    for key, value in element.attrib.items():
        attr_ns_uri, attr_local_name = get_namespace_info(key)
        clean_key = format_key(attr_local_name)
        
        # Add namespace info for attribute if present
        if attr_ns_uri:
            attr_ns_prefix = get_namespace_prefix(attr_ns_uri, namespaces)
            if attr_ns_prefix:
                attributes[clean_key] = {
                    "_namespace": attr_ns_prefix,
                    "value": format_value(value)
                }
        else:
            attributes[clean_key] = format_value(value)
    
    if attributes:
        result["_attributes"] = attributes
    
    # Process child elements
    child_elements = list(element)
    
    # If there are no child elements, just return the text content or current result
    if not child_elements:
        if element.text and element.text.strip():
            result["_value"] = format_value(element.text.strip())
            return result
        return result
    
    # Process child elements
    for child in child_elements:
        # Get namespace and tag name
        child_ns_uri, child_local_name = get_namespace_info(child.tag)
        clean_tag = format_key(child_local_name)
        
        child_dict = element_to_clean_dict(child, namespaces)
        
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
        result["_content"] = format_value(element.text.strip())
        
    return result

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
