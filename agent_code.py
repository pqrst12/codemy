# Function to call LLM with required inputs
def call_llm(chat_history, user_input, agent_scratchpad, user_question):
    # Call the LLM with the given inputs and return the LLM's response
    return llm_response

# Main agent loop
def agent_loop(user_input):
    chat_history = []
    agent_scratchpad = []  # Initial empty scratchpad
    
    user_question = user_input
    
    while True:
        llm_response = call_llm(chat_history, user_input, agent_scratchpad, user_question)
        
        # Check if LLM decides to use a tool
        if llm_response['thought'] == "do i need to use tool?yes":
            tool_name = llm_response['action']
            tool_input = llm_response['action input']
            
            # Execute the tool and get observation
            observation = execute_tool(tool_name, tool_input)
            
            # Update the agent scratchpad with tool usage and observation
            agent_scratchpad.append({
                'thought': llm_response['thought'],
                'action': tool_name,
                'action input': tool_input,
                'observation': observation
            })
        
        # Check if LLM gives a final answer
        elif llm_response['thought'] == "Do I need to use a tool? No":
            final_answer = llm_response['final_answer']
            
            # Append to chat history
            chat_history.append({
                'question': user_question,
                'answer': final_answer
            })
            
            # Terminate the loop
            break
    
    # Return the chat history (including final response)
    return chat_history
