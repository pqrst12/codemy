import xml.etree.ElementTree as ET
import os

def split_xml(file_path, output_dir, max_nodes_per_file=2):
    """Parses an XML file and splits it into multiple text files, preserving nested structure."""
    
    # Parse XML using ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all first-level child elements (excluding text nodes)
    child_elements = [elem for elem in root if isinstance(elem.tag, str)]

    # Process child elements in chunks
    for i in range(0, len(child_elements), max_nodes_per_file):
        chunk = child_elements[i : i + max_nodes_per_file]

        # Create a new XML structure with the same root
        new_root = ET.Element(root.tag, root.attrib)

        # Copy namespace if present
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0].strip("{")
            new_root.set("xmlns", namespace)

        # Add the selected child nodes
        for child in chunk:
            new_root.append(child)

        # Convert to formatted XML text
        output_text = format_xml(new_root)

        # Write to text file
        output_file = os.path.join(output_dir, f"split_part_{i//max_nodes_per_file + 1}.xml")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"Saved: {output_file}")

def format_xml(element):
    """Returns a pretty-printed XML string from an ElementTree Element."""
    return ET.tostring(element, encoding="utf-8").decode("utf-8")

# Example usage
xml_file = "complex_sample.xml"  # Replace with your XML file
output_directory = "split_text_files"
split_xml(xml_file, output_directory, max_nodes_per_file=1)
