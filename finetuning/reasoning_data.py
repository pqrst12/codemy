# Import necessary libraries
import os
import random
import json
from tqdm.notebook import tqdm  # Use notebook version of tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import traceback
import copy
import time

# Placeholder for API interaction
# Replace this with your actual implementation or API client
def call_llm_api(content, model_name="your-model-name", additional_args={"max_tokens": 8192}):
    """
    Function to call your preferred LLM API
    
    Replace this implementation with actual API calls to your model of choice
    """
    print("Calling API...")
    time.sleep(1)  # Simulating API call delay
    # In a real implementation, you would call your API here
    return "Your API response would appear here"

# Placeholder for prompts - replace these with your actual prompts
PROMPT_TEMPLATES = {
    "verify": "Verification prompt template goes here",
    "init": "Initial reasoning prompt template goes here",
    "backtracking": "Backtracking prompt template goes here",
    "explore_new_path": "Explore new path prompt template goes here",
    "verification": "Verification-focused prompt template goes here",
    "correction": "Correction prompt template goes here",
    "with_label": "With label prompt template goes here",
    "reformat": "Reformat prompt template goes here",
    "get_final": "Get final response prompt template goes here"
}

# Define search strategies
SEARCH_STRATEGIES = [
    ('Backtracking', PROMPT_TEMPLATES["backtracking"]),
    ('Exploring New Paths', PROMPT_TEMPLATES["explore_new_path"]),
    ('Verification', PROMPT_TEMPLATES["verification"]),
    ('Correction', PROMPT_TEMPLATES["correction"])
]

# Helper functions for parsing responses
def extract_bracket_content(text):
    """Extract content between the first '{' and the last '}'"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def parse_cot_response(response):
    """Parse structured CoT response from LLM"""
    try:
        if response and '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))
        
        # Validate structure
        assert isinstance(da["CoT"], list), "CoT should be list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        
        return True, da
    except Exception as e:
        print(f"Error parsing response: {e}")
        traceback.print_exc()
        return False, None

def parse_reformatted_response(response):
    """Parse reformatted natural reasoning response"""
    try:
        if response and '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n',''))
        
        # Validate structure
        assert isinstance(da["NaturalReasoning"], str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have line breaks"
        
        return True, da
    except Exception as e:
        print(f"Error parsing reformatted response: {e}")
        traceback.print_exc()
        return False, None

def format_cot_as_text(longcot):
    """Format CoT steps as readable text"""
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'], x['content']))
        else:
            resstr.append(temp.format(
                x['action'].replace('Final Conclusion', 'Conclusion'), 
                x['content']
            ))
    return '\n'.join(resstr).strip()

# Main processing function
def process_single_question(question_data, config):
    """
    Process a single question through the reasoning path search
    
    Args:
        question_data: Dict containing the question data
        config: Configuration dictionary for processing
    
    Returns:
        Updated question_data with reasoning paths
    """
    try:
        retry_time = 1
        d = question_data.copy()
        
        # Initialize tracking data
        d['verify'] = []
        d['Long_CoT'] = []
        d['api_queries'] = []
        d['api_responses'] = []
        d['response_struct'] = []
        d['response_type'] = []
        d['prior_fail_try'] = []
        
        question = d['Open-ended Verifiable Question']
        ground_truth = d['Ground-True Answer']
        
        # Step 1: Initial reasoning
        init_query = PROMPT_TEMPLATES["init"].format(question)
        d['api_queries'].append(init_query)
        
        # Try getting valid initial reasoning
        for _ in range(retry_time):
            init_response = call_llm_api(init_query, model_name=config["model_name"])
            d['api_responses'].append(init_response)
            
            flag, struct = parse_cot_response(init_response)
            if flag:
                d['response_struct'].append(struct["CoT"])
                d['Long_CoT'] = struct["CoT"]
                d['response_type'].append('Init_CoT')
                break
            else:
                print('Retrying Init_CoT')
        
        if not flag:
            raise Exception('Initial reasoning failed')
        
        # Verify against ground truth
        verify_query = PROMPT_TEMPLATES["verify"].format(d['Long_CoT'][-2]['content'], ground_truth)
        d['api_queries'].append(verify_query)
        verify_response = call_llm_api(verify_query, model_name=config["model_name"])
        d['api_responses'].append(verify_response)
        
        # Update verification status
        is_correct = 'true' in verify_response.lower()
        d['verify'].append(is_correct)
        
        # Step 2: Iterative search if needed
        for attempt in range(config["max_search_attempts"]):
            if attempt > 0:
                # Archive the failed state
                if 'prior_fail_try' not in d:
                    d['prior_fail_try'] = []
                d['prior_fail_try'].append(d.copy())
                # Reset to earlier state
                d = question_data.copy()
            
            # Begin search iterations
            for search_iter in range(config["max_search_depth"]):
                if d['verify'][-1]:  # If already correct, break
                    break
                
                # Format current reasoning path
                reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
                
                # Choose strategy - first iteration avoids backtracking
                if search_iter > 0:
                    strategy_name, strategy_prompt = random.choice(SEARCH_STRATEGIES)
                else:
                    strategy_name, strategy_prompt = random.choice(SEARCH_STRATEGIES[1:])
                
                # Format query with chosen strategy
                query = strategy_prompt.format(question, reasoning)
                d['api_queries'].append(query)
                
                # Try getting valid response with retry
                flag = False
                for _ in range(retry_time):
                    response = call_llm_api(query, model_name=config["model_name"])
                    flag, struct = parse_cot_response(response)
                    
                    if flag:
                        d['api_responses'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append(f'Re_CoT_{strategy_name}')
                        break
                    else:
                        print(f'Retrying strategy {strategy_name}')
                
                if not flag:
                    raise Exception(f'Rethinking with {strategy_name} failed')
                
                # Verify again
                verify_query = PROMPT_TEMPLATES["verify"].format(d['Long_CoT'][-2]['content'], ground_truth)
                d['api_queries'].append(verify_query)
                verify_response = call_llm_api(verify_query, model_name=config["model_name"])
                d['api_responses'].append(verify_response)
                
                is_correct = 'true' in verify_response.lower()
                d['verify'].append(is_correct)
            
            if d['verify'][-1]:  # If correct, break out of attempts
                break
        
        # Step 3: If still incorrect and efficient search is enabled, use labeled approach
        if not d['verify'][-1] and config["efficient_search"]:
            reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
            query = PROMPT_TEMPLATES["with_label"].format(question, reasoning, ground_truth)
            d['api_queries'].append(query)
            
            for _ in range(retry_time):
                response = call_llm_api(query, model_name=config["model_name"])
                flag, struct = parse_cot_response(response)
                
                if flag:
                    d['api_responses'].append(response)
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                    d['response_type'].append('Label_CoT')
                    # Assume correct with label
                    d['verify'].append(True)
                    break
                else:
                    print('Retrying Label_CoT')
            
            if not flag:
                raise Exception('Label-guided reasoning failed')
        
        # Step 4: Generate final natural reasoning and response
        if d['verify'][-1]:
            # Format reasoning as text
            reasoning_text = format_cot_as_text(d['Long_CoT'])
            
            # Reformat to natural reasoning
            reformat_query = PROMPT_TEMPLATES["reformat"].format(reasoning_text, question)
            d['api_queries'].append(reformat_query)
            
            for _ in range(retry_time):
                reformat_response = call_llm_api(reformat_query, model_name=config["model_name"])
                flag, struct = parse_reformatted_response(reformat_response)
                
                if flag:
                    d['api_responses'].append(reformat_response)
                    d["Complex_CoT"] = struct["NaturalReasoning"]
                    
                    # Generate final user-facing response
                    final_query = PROMPT_TEMPLATES["get_final"].format(
                        d['Complex_CoT'], question
                    )
                    d['api_queries'].append(final_query)
                    final_response = call_llm_api(final_query, model_name=config["model_name"])
                    d['api_responses'].append(final_response)
                    d["Response"] = final_response
                    d["Question"] = question
                    break
                else:
                    print('Retrying reformatting')
        
        return d
    
    except Exception as e:
        print(f"Error processing question: {e}")
        traceback.print_exc()
        return None

# Functions for loading and saving data
def load_data(data_path, limit=None):
    """Load and filter data from JSON file"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Add process IDs
    tmp_id = 1
    for item in data:
        item['process_id'] = tmp_id
        tmp_id += 1
    
    # Filter data to only include required fields
    filtered_data = []
    for item in data:
        if 'Open-ended Verifiable Question' in item and 'Ground-True Answer' in item:
            filtered_data.append(item)
    
    print(f"Original data size: {len(data)}, Filtered data size: {len(filtered_data)}")
    
    # Apply limit if specified
    if limit:
        filtered_data = filtered_data[:limit]
    
    return filtered_data

def deduplicate_data(data, processed_data):
    """Remove already processed items from data"""
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]

def merge_saved_files(save_dir):
    """Merge saved JSON files from a directory"""
    _, _, filenames = [i for i in os.walk(save_dir)][0]
    json_files = [f for f in filenames if f.endswith('.json')]
    res = []
    
    for file_path in json_files:
        try:
            with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                da = json.loads(f.read())
                if 'Complex_CoT' in da and 'Response' in da:
                    res.append(da)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return res

def save_results(item, save_dir):
    """Save a single result item to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{item['process_id']}.json")
    
    with open(save_path, mode="w", encoding="utf-8") as fw:
        json.dump(item, fw, ensure_ascii=False, indent=2)
    
    return True

# Interactive functions for Jupyter notebook
def process_batch(data, config, save_dir=None):
    """Process a batch of questions with progress tracking"""
    results = []
    
    for item in tqdm(data, desc="Processing questions"):
        result = process_single_question(item, config)
        if result and save_dir:
            save_results(result, save_dir)
        if result:
            results.append(result)
    
    return results

def run_parallel_processing(data, config, save_dir=None, max_workers=4):
    """Process questions in parallel using ThreadPoolExecutor"""
    results = []
    os.makedirs(save_dir, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_single_question, item, config): item 
            for item in data
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_item), 
            total=len(data), 
            desc="Processing in parallel"
        ):
            item = future_to_item[future]
            try:
                result = future.result()
                if result and save_dir:
                    save_results(result, save_dir)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing item {item['process_id']}: {e}")
    
    return results

def create_config(
    model_name="your-model-name",
    max_search_attempts=1,
    max_search_depth=2,
    efficient_search=True,
    num_process=5
):
    """Create a configuration dictionary"""
    return {
        "model_name": model_name,
        "max_search_attempts": max_search_attempts,
        "max_search_depth": max_search_depth,
        "efficient_search": efficient_search,
        "num_process": num_process
    }

# Main execution function
def main(
    data_path, 
    save_dir, 
    model_name="your-model-name", 
    limit=None, 
    parallel=False,
    max_workers=None
):
    """
    Main execution function for the reasoning path search
    
    Args:
        data_path: Path to the input data JSON file
        save_dir: Directory to save the processing results
        model_name: Name of the LLM model to use
        limit: Optional limit on the number of questions to process
        parallel: Whether to use parallel processing
        max_workers: Number of workers for parallel processing
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_path, limit=limit)
    print(f"Loaded {len(data)} questions for processing")
    
    # Check for existing processed data
    processed_data = []
    if os.path.exists(save_dir):
        processed_data = merge_saved_files(save_dir)
        print(f"Found {len(processed_data)} already processed items")
    
    # Remove already processed items
    data = deduplicate_data(data, processed_data)
    print(f"Remaining items to process: {len(data)}")
    
    if not data:
        print("All items have been processed already.")
        return processed_data
    
    # Create configuration
    config = create_config(
        model_name=model_name,
        max_search_attempts=1,
        max_search_depth=2,
        efficient_search=True,
        num_process=max_workers if max_workers else multiprocessing.cpu_count()
    )
    
    # Process data
    if parallel and len(data) > 1:
        if not max_workers:
            max_workers = min(multiprocessing.cpu_count(), 8)  # Default to 8 or CPU count
        print(f"Processing in parallel with {max_workers} workers")
        results = run_parallel_processing(data, config, save_dir, max_workers=max_workers)
    else:
        print("Processing sequentially")
        results = process_batch(data, config, save_dir)
    
    # Combine with previously processed data
    all_results = processed_data + results
    print(f"Total processed items: {len(all_results)}")
    
    # Save combined results
    combined_results_path = os.path.join(save_dir, "combined_results.json")
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    return all_results

# Analysis functions
def analyze_results(results):
    """
    Analyze the results of the reasoning path search
    
    Args:
        results: List of processed question results
    
    Returns:
        Dictionary with analysis metrics
    """
    total = len(results)
    correct = sum(1 for r in results if r['verify'][-1])
    
    strategy_counts = {}
    for r in results:
        for strategy in r['response_type']:
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            strategy_counts[strategy] += 1
    
    avg_iterations = sum(len(r['verify']) for r in results) / total if total > 0 else 0
    
    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": correct / total if total > 0 else 0,
        "strategy_usage": strategy_counts,
        "avg_iterations": avg_iterations
    }

def print_example_reasoning(results, index=0):
    """
    Print an example reasoning path
    
    Args:
        results: List of processed question results
        index: Index of the result to print
    """
    if not results or index >= len(results):
        print("No results available or invalid index")
        return
    
    r = results[index]
    
    print(f"Question: {r['Question']}")
    print(f"Ground Truth: {r['Ground-True Answer']}")
    print(f"Final Answer Correct: {r['verify'][-1]}")
    print("\nReasoning Process:")
    
    for i, step_type in enumerate(r['response_type']):
        print(f"\n--- Step {i+1}: {step_type} ---")
        if i < len(r['response_struct']):
            for step in r['response_struct'][i]:
                if 'action' in step:
                    print(f"\n{step['action']}:")
                    print(step['content'])
    
    print("\nFinal Natural Reasoning:")
    print(r.get('Complex_CoT', 'Not available'))
    
    print("\nFinal Response:")
    print(r.get('Response', 'Not available'))

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reasoning path search on a dataset of questions")
    parser.add_argument("--data", type=str, required=True, help="Path to input data JSON file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model", type=str, default="your-model-name", help="Model name to use")
    parser.add_argument("--limit", type=int, help="Limit the number of questions to process")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    results = main(
        args.data,
        args.save_dir,
        model_name=args.model,
        limit=args.limit,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Print analysis
    analysis = analyze_results(results)
    print("\nResults Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Print an example
    if results:
        print("\nExample Processing Result:")
        print_example_reasoning(results)
