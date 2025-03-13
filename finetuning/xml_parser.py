import xml.etree.ElementTree as ET
import os
from xml.dom import minidom

def split_xml(file_path, output_dir, max_nodes_per_file=2):
    """Parses and splits an XML file into multiple text files while retaining structure."""
    
    # Parse XML using minidom to handle comments & processing instructions
    dom_tree = minidom.parse(file_path)
    root = dom_tree.documentElement  # Get root element

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract child elements
    child_elements = [node for node in root.childNodes if node.nodeType == node.ELEMENT_NODE]

    # Process in chunks
    for i in range(0, len(child_elements), max_nodes_per_file):
        chunk = child_elements[i : i + max_nodes_per_file]

        # Create new XML document
        new_doc = minidom.Document()
        new_root = new_doc.createElement(root.tagName)

        # Copy attributes if any
        for attr_name, attr_value in root.attributes.items():
            new_root.setAttribute(attr_name, attr_value)

        new_doc.appendChild(new_root)

        # Copy comments, processing instructions, and elements
        for node in root.childNodes:
            if node in chunk or node.nodeType in {node.COMMENT_NODE, node.PROCESSING_INSTRUCTION_NODE}:
                new_root.appendChild(node.cloneNode(True))

        # Write to file
        output_file = os.path.join(output_dir, f"split_part_{i//max_nodes_per_file + 1}.xml")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(new_doc.toprettyxml(indent="  "))

        print(f"Saved: {output_file}")

# Example usage
xml_file = "complex_sample.xml"  # Replace with your XML file
output_directory = "split_text_files"
split_xml(xml_file, output_directory, max_nodes_per_file=1)
