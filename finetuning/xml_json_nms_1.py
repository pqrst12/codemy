import xml.etree.ElementTree as ET
import json
import os
import glob

def xml_to_json(xml_file_path, json_file_path=None):
    if json_file_path is None:
        json_file_path = os.path.splitext(xml_file_path)[0] + '.json'
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract namespaces only at the root level
    namespaces = extract_namespaces(root)
    
    json_data = {
        "namespaces": namespaces,
        "content": element_to_dict(root, namespaces, is_root=True)
    }
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {xml_file_path} to {json_file_path}")
    return json_file_path

def extract_namespaces(element):
    namespaces = {}
    if hasattr(element, 'attrib'):
        for attr_name, value in element.attrib.items():
            if attr_name.startswith('xmlns:'):
                prefix = attr_name.split(':', 1)[1]
                namespaces[prefix] = value
            elif attr_name == 'xmlns':
                namespaces["default"] = value
    return namespaces

def get_namespace_prefix(tag, namespaces):
    if '}' in tag:
        ns_uri, local_name = tag[1:].split('}', 1)
        for prefix, uri in namespaces.items():
            if uri == ns_uri:
                return prefix, local_name
        return None, local_name
    else:
        return None, tag

def element_to_dict(element, namespaces, is_root=False):
    prefix, local_name = get_namespace_prefix(element.tag, namespaces)
    
    result = {
        "name": local_name,
        "attributes": {},
        "children": []
    }
    
    # Only add namespace info at the root level
    if is_root and prefix:
        result["namespace"] = prefix
    
    for attr_name, attr_value in element.attrib.items():
        _, attr_local_name = get_namespace_prefix(attr_name, namespaces)
        result["attributes"][attr_local_name] = format_value(attr_value)
    
    if element.text and element.text.strip():
        result["text"] = format_value(element.text.strip())
    
    for child in element:
        child_dict = element_to_dict(child, namespaces, is_root=False)
        result["children"].append(child_dict)
    
    if not result["children"]:
        del result["children"]
    
    if not result["attributes"]:
        del result["attributes"]
    
    return result

def format_value(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ('true', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'no', 'n', '0'):
        return False
    return value

def convert_all_xml_files(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    return [xml_to_json(xml_file, os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.json')) for xml_file in xml_files]

if __name__ == "__main__":
    xml_file = "split_text_files/split_part_1.xml"
    xml_to_json(xml_file)
    input_directory = "split_text_files"
    output_directory = "json_output"
    convert_all_xml_files(input_directory, output_directory)
