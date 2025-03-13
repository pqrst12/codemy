import xml.etree.ElementTree as ET
import os
import copy
from xml.dom import minidom

def split_xml(file_path, output_dir, max_nodes_per_file=50, large_element_threshold=20):
    """
    Parses an XML file and splits it into multiple files while preserving nested structure.
    Further splits large elements (with more than large_element_threshold children) into chunks.
    """
    # Parse XML using ElementTree
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all elements with large number of children
    large_elements = []
    for elem in root.iter():
        if len(list(elem)) > large_element_threshold:
            large_elements.append(elem)
    
    file_counter = 1
    
    # Process large elements first
    for large_elem in large_elements:
        # Get the path to this large element
        path_to_large = get_element_path(root, large_elem)
        
        if not path_to_large:
            continue
            
        # Split the children of this large element
        children = list(large_elem)
        for i in range(0, len(children), max_nodes_per_file):
            chunk = children[i:i+max_nodes_per_file]
            
            # Create a new XML structure with the same root
            new_root = ET.Element(root.tag, root.attrib)
            
            # Copy namespace if present
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0].strip("{")
                new_root.set("xmlns", namespace)
            
            # Recreate the path to the large element
            current = new_root
            for path_elem in path_to_large[:-1]:  # All except the large element itself
                new_elem = ET.SubElement(current, path_elem.tag, path_elem.attrib)
                if path_elem.text and path_elem.text.strip():
                    new_elem.text = path_elem.text
                current = new_elem
            
            # Create the large element
            large_elem_copy = ET.SubElement(current, large_elem.tag, large_elem.attrib)
            if large_elem.text and large_elem.text.strip():
                large_elem_copy.text = large_elem.text
            
            # Add the chunk of children
            for child in chunk:
                add_element_with_children(large_elem_copy, child)
            
            # Convert to formatted XML text
            output_text = format_xml(new_root)
            
            # Write to text file
            output_file = os.path.join(output_dir, f"split_part_{file_counter}.xml")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            print(f"Saved: {output_file}")
            file_counter += 1
    
    # Now handle the remaining structure (excluding already processed large elements)
    processed_elements = set(large_elements)
    
    # Check if we still have top-level elements to process
    remaining_top_level = [elem for elem in root if elem not in processed_elements]
    
    if remaining_top_level:
        # Process remaining top-level elements
        for i in range(0, len(remaining_top_level), max_nodes_per_file):
            chunk = remaining_top_level[i:i+max_nodes_per_file]
            
            # Create a new XML structure with the same root
            new_root = ET.Element(root.tag, root.attrib)
            
            # Copy namespace if present
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0].strip("{")
                new_root.set("xmlns", namespace)
            
            # Add the selected child nodes (with deep copy to preserve structure)
            for child in chunk:
                # Skip if this is a large element or contains large elements
                if child in processed_elements or contains_any(child, processed_elements):
                    continue
                
                new_child = copy.deepcopy(child)
                new_root.append(new_child)
            
            # Only save if there are elements to save
            if len(new_root):
                # Convert to formatted XML text
                output_text = format_xml(new_root)
                
                # Write to text file
                output_file = os.path.join(output_dir, f"split_part_{file_counter}.xml")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"Saved: {output_file}")
                file_counter += 1
    else:
        # If no top-level elements or all were processed as large elements,
        # check if we've processed anything at all
        if not large_elements:
            # No large elements were found, so we'll just split the entire XML
            # Get all elements at any level
            all_elements = []
            for elem in root.iter():
                if elem != root and elem not in processed_elements:
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
                    # Skip if this is a large element or is contained in a large element
                    if elem in processed_elements or any(contains(le, elem) for le in large_elements):
                        continue
                        
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
                
                # Only save if there are elements to save
                if len(list(new_root)):
                    # Convert to formatted XML text
                    output_text = format_xml(new_root)
                    
                    # Write to text file
                    output_file = os.path.join(output_dir, f"split_part_{file_counter}.xml")
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(output_text)
                    print(f"Saved: {output_file}")
                    file_counter += 1

def contains(parent, child):
    """Check if parent contains child element."""
    if parent == child:
        return True
    
    for elem in parent.iter():
        if elem == child:
            return True
    
    return False

def contains_any(parent, elements):
    """Check if parent contains any of the elements in the set."""
    for elem in elements:
        if contains(parent, elem):
            return True
    return False

def get_element_path(root, target_elem):
    """Find the path from root to the target element."""
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
split_xml(xml_file, output_directory, max_nodes_per_file=50, large_element_threshold=20)
