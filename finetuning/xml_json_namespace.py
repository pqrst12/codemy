import xml.etree.ElementTree as ET
import json
import os
import glob
import re

def xml_to_json(xml_file_path, json_file_path=None):
    """
    Convert an XML file to JSON format, preserving namespaces.
    
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
    
    # Extract namespace information
    namespaces = extract_namespaces(root)
    
    # Convert XML to JSON
    json_data = {
        "namespaces": namespaces,
        "content": element_to_dict(root, namespaces)
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
            for prefix, uri in element.nsmap.items() if hasattr(element, 'nsmap') else []:
                if uri == ns_uri:
                    namespaces[prefix if prefix else "default"] = uri
        
        # Check attributes for namespaces
        for attr_name in elem.attrib:
            if '}' in attr_name:
                ns_uri = attr_name.split('}', 1)[0].strip('{')
                for prefix, uri in element.nsmap.items() if hasattr(element, 'nsmap') else []:
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

def get_namespace_prefix(tag, namespaces):
    """
    Get namespace prefix from a tag.
    
    Args:
        tag (str): The tag with possible namespace
        namespaces (dict): Dictionary of namespace prefixes and URIs
        
    Returns:
        tuple: (prefix, local_name) or (None, tag) if no namespace
    """
    if '}' in tag:
        ns_uri = tag.split('}', 1)[0].strip('{')
        local_name = tag.split('}', 1)[1]
        
        # Find the prefix for this namespace URI
        for prefix, uri in namespaces.items():
            if uri == ns_uri:
                return prefix, local_name
                
        # If no prefix found, use the URI as prefix
        return ns_uri, local_name
    else:
        return None, tag

def element_to_dict(element, namespaces):
    """
    Convert an XML element to a dictionary with namespace information.
    
    Args:
        element: An ElementTree element
        namespaces (dict): Dictionary of namespace prefixes and URIs
        
    Returns:
        dict: Dictionary representation of the element
    """
    # Get the element's tag with namespace prefix
    prefix, local_name = get_namespace_prefix(element.tag, namespaces)
    
    result = {
        "name": local_name,
        "attributes": {},
        "children": []
    }
    
    if prefix:
        result["namespace"] = prefix
    
    # Add element attributes
    for attr_name, attr_value in element.attrib.items():
        attr_prefix, attr_local_name = get_namespace_prefix(attr_name, namespaces)
        attr_obj = {
            "name": attr_local_name,
            "value": format_value(attr_value)
        }
        if attr_prefix:
            attr_obj["namespace"] = attr_prefix
        result["attributes"][attr_local_name] = attr_obj
    
    # Add element text if it exists and is not just whitespace
    if element.text and element.text.strip():
        result["text"] = format_value(element.text.strip())
    
    # Process child elements
    for child in element:
        child_dict = element_to_dict(child, namespaces)
        result["children"].append(child_dict)
    
    # If there are no children, remove the empty children array
    if not result["children"]:
        del result["children"]
    
    # If there are no attributes, remove the empty attributes dictionary
    if not result["attributes"]:
        del result["attributes"]
    
    return result

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
    Convert all XML files in a directory to JSON format.
    
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
