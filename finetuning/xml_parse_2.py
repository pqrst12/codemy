import xml.etree.ElementTree as ET
import os
import copy
from xml.dom import minidom

def split_xml(file_path, output_dir, max_nodes_per_file=50):
    """Parses an XML file and splits it into multiple files while preserving nested structure."""
    
    # Parse XML using ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle XML with deeply nested structure
    if len(list(root)) <= 1:
        # Find all elements at any level that have children
        container_elements = []
        for elem in root.iter():
            if len(list(elem)) > 0 and elem != root:
                container_elements.append(elem)
        
        # If we found container elements, split by them
        if container_elements:
            # Process container elements in chunks
            for i in range(0, len(container_elements), max_nodes_per_file):
                chunk = container_elements[i:i+max_nodes_per_file]
                
                # Create a new XML structure with the same root
                new_root = ET.Element(root.tag, root.attrib)
                
                # Copy namespace if present
                if root.tag.startswith("{"):
                    namespace = root.tag.split("}")[0].strip("{")
                    new_root.set("xmlns", namespace)
                
                # For each container element, create a path to it
                for container in chunk:
                    # Find the path from root to this container
                    path_elements = get_element_path(root, container)
                    
                    if path_elements:
                        # Create the path in the new XML
                        current_new = new_root
                        for path_elem in path_elements[:-1]:  # All but the last element (container itself)
                            # Find or create the parent element
                            tag = path_elem.tag
                            attrib = path_elem.attrib
                            
                            # Check if this parent already exists in our new structure
                            existing = None
                            for child in current_new:
                                if child.tag == tag and child.attrib == attrib:
                                    existing = child
                                    break
                            
                            if existing is None:
                                # Create new parent element
                                new_elem = ET.SubElement(current_new, tag, attrib)
                                if path_elem.text and path_elem.text.strip():
                                    new_elem.text = path_elem.text
                                current_new = new_elem
                            else:
                                current_new = existing
                        
                        # Add the container element itself with all its children
                        add_element_with_children(current_new, path_elements[-1])
                
                # Convert to formatted XML text
                output_text = format_xml(new_root)
                
                # Write to text file
                output_file = os.path.join(output_dir, f"split_part_{i//max_nodes_per_file + 1}.xml")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"Saved: {output_file}")
        else:
            # If no container elements found, split by individual elements
            all_elements = []
            for elem in root.iter():
                if elem != root:
                    all_elements.append(elem)
            
            # Process elements in chunks
            for i in range(0, len(all_elements), max_nodes_per_file):
                chunk = all_elements[i:i+max_nodes_per_file]
                
                # Create a new XML structure with the same root
                new_root = ET.Element(root.tag, root.attrib)
                
                # Copy namespace if present
                if root.tag.startswith("{"):
                    namespace = root.tag.split("}")[0].strip("{")
                    new_root.set("xmlns", namespace)
                
                # Create a structure that mimics the original
                element_dict = {}
                for elem in chunk:
                    path_elements = get_element_path(root, elem)
                    if path_elements:
                        current_new = new_root
                        for path_elem in path_elements[:-1]:
                            tag = path_elem.tag
                            attrib_tuple = tuple(sorted(path_elem.attrib.items()))
                            
                            # Create a unique key for this element
                            key = (tag, attrib_tuple, id(path_elem))
                            
                            if key in element_dict:
                                current_new = element_dict[key]
                            else:
                                new_elem = ET.SubElement(current_new, tag, path_elem.attrib)
                                if path_elem.text and path_elem.text.strip():
                                    new_elem.text = path_elem.text
                                element_dict[key] = new_elem
                                current_new = new_elem
                        
                        # Add the element itself
                        last_elem = path_elements[-1]
                        new_elem = ET.SubElement(current_new, last_elem.tag, last_elem.attrib)
                        if last_elem.text and last_elem.text.strip():
                            new_elem.text = last_elem.text
                
                # Convert to formatted XML text
                output_text = format_xml(new_root)
                
                # Write to text file
                output_file = os.path.join(output_dir, f"split_part_{i//max_nodes_per_file + 1}.xml")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"Saved: {output_file}")
    else:
        # If we have multiple top-level elements, use a simpler approach
        top_level_elements = list(root)
        for i in range(0, len(top_level_elements), max_nodes_per_file):
            chunk = top_level_elements[i:i+max_nodes_per_file]
            
            # Create a new XML structure with the same root
            new_root = ET.Element(root.tag, root.attrib)
            
            # Copy namespace if present
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0].strip("{")
                new_root.set("xmlns", namespace)
            
            # Add the selected child nodes (with deep copy to preserve structure)
            for child in chunk:
                new_child = copy.deepcopy(child)
                new_root.append(new_child)
            
            # Convert to formatted XML text
            output_text = format_xml(new_root)
            
            # Write to text file
            output_file = os.path.join(output_dir, f"split_part_{i//max_nodes_per_file + 1}.xml")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Saved: {output_file}")

def get_element_path(root, target_elem):
    """Find the path from root to the target element."""
    path = []
    for elem in root.iter():
        if elem == target_elem:
            return path + [elem]
        if target_elem in elem:
            path.append(elem)
    
    # Try a more thorough search
    def find_path(current, target, current_path):
        if current == target:
            return current_path + [current]
        
        for child in current:
            result = find_path(child, target, current_path + [current])
            if result:
                return result
        
        return None
    
    return find_path(root, target_elem, [])

def add_element_with_children(parent, element):
    """Add an element and all its children to the parent."""
    new_elem = ET.SubElement(parent, element.tag, element.attrib)
    if element.text and element.text.strip():
        new_elem.text = element.text
    
    for child in element:
        add_element_with_children(new_elem, child)

def format_xml(element):
    """Returns a pretty-printed XML string from an ElementTree Element."""
    rough_string = ET.tostring(element, encoding="utf-8")
    
    # Use minidom for pretty printing
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Example usage
xml_file = "complex_sample.xml"  # Replace with your XML file
output_directory = "split_text_files"
split_xml(xml_file, output_directory, max_nodes_per_file=50)
