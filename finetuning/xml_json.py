import xml.etree.ElementTree as ET
import json
import os
import glob

def xml_to_json(xml_file_path, json_file_path=None):
    """
    Convert an XML file to JSON format.
    
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
    
    # Convert XML to JSON
    json_data = element_to_dict(root)
    
    # Write JSON to file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {xml_file_path} to {json_file_path}")
    return json_file_path

def element_to_dict(element):
    """
    Convert an XML element to a dictionary representation.
    
    Args:
        element: An ElementTree element
    
    Returns:
        dict: Dictionary representation of the element
    """
    result = {}
    
    # Add element attributes
    if element.attrib:
        result["@attributes"] = dict(element.attrib)
    
    # Add element text if it exists and is not just whitespace
    if element.text and element.text.strip():
        result["#text"] = element.text.strip()
    
    # Process child elements
    for child in element:
        child_dict = element_to_dict(child)
        
        # Handle child elements with the same tag
        if child.tag in result:
            if type(result[child.tag]) is list:
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict
    
    return result

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
