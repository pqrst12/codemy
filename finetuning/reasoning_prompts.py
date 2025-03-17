# %%

import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import random
import requests
from retrying import retry
import argparse
import re
import traceback
import copy

class GPT:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"Using model: {self.model_name}")

    def call(self, content, additional_args={}):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **additional_args,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response_data = response.json()

        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data}")

        return response_data['choices'][0]['message']['content']

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)

verify_prompt = """<Model Response>  
{}  
</Model Response>  

<Reference Resolution>  
{}
</Reference Resolution>  

You are provided with a model-generated resolution (<Model Response>) and a reference resolution (<Reference Resolution>). Compare the model resolution with the reference resolution and determine if it correctly addresses the customer issue with an appropriate solution. Your task is to simply output "True" if the response is correct and addresses the same core issues as the reference, and "False" otherwise."""


query_prompt_init = """<customer ticket>
{}
</customer ticket>

Please respond to the above customer ticket using the Chain of Thought (CoT) reasoning method. Your response should consist of multiple steps, each of which includes three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

- **'Inner Thinking'**: This is the step where thinking is done. Note that multiple 'Inner Thinking' steps are required to describe thorough reasoning. Each step should first generate a brief title.
- **'Final Resolution'**: At this stage, you summarize the correct reasoning from previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is required here.
- **'Verification'**: At this stage, you verify the resolution from the "Final Resolution" step. If the resolution addresses all aspects of the customer's issue, end the process. If not, return to "Inner Thinking" for further reasoning. No title is required here.

The output format must strictly follow the JSON structure below:
```json
{{
"CoT": [
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Backtracking = """<customer ticket>
{}
</customer ticket>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Resolution**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is needed for this section.
3. **Verification**: Verify the accuracy and completeness of the "Final Resolution". If it fully addresses the customer's issue, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<customer ticket> represents the customer issue to be resolved, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Resolution** is incomplete or incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning using **backtracking** to revisit earlier points of reasoning and construct a new Final Resolution.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Exploring_New_Path = """<customer ticket>
{}
</customer ticket>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Resolution**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is needed for this section.
3. **Verification**: Verify the accuracy and completeness of the "Final Resolution". If it fully addresses the customer's issue, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<customer ticket> represents the customer issue to be resolved, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Resolution** is incomplete or incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by exploring new approaches to resolving this customer issue and construct a new Final Resolution.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Verification = """<customer ticket>
{}
</customer ticket>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Resolution**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is needed for this section.
3. **Verification**: Verify the accuracy and completeness of the "Final Resolution". If it fully addresses the customer's issue, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<customer ticket> represents the customer issue to be resolved, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Resolution** is incomplete or incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by conducting a thorough **validation** process to ensure completeness and appropriateness of the resolution and construct a new Final Resolution.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_rethink_Correction = """<customer ticket>
{}
</customer ticket>

<previous reasoning>
{}
<previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Resolution**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is needed for this section.
3. **Verification**: Verify the accuracy and completeness of the "Final Resolution". If it fully addresses the customer's issue, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<customer ticket> represents the customer issue to be resolved, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. I have manually reviewed the reasoning and determined that the **Final Resolution** is incomplete or incorrect. Your 'Verification' results must align with mine. Proceed to refine the reasoning by making precise **corrections** to address prior flaws and construct a new Final Resolution.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

gen_prompt_w_label = """<customer ticket>
{}
</customer ticket>

<previous reasoning>
{}
</previous reasoning>

<response requirements>
Your response must include the following steps, each composed of three types of actions: **"Inner Thinking"**, **"Final Resolution"**, and **"Verification"**:

1. **Inner Thinking**: Break down the reasoning process into multiple concise steps. Each step should start with a brief title to clarify its purpose.
2. **Final Resolution**: Summarize the correct reasoning from all previous 'Inner Thinking' steps and provide the final resolution to the customer's issue. No title is needed for this section.
3. **Verification**: Verify the accuracy and completeness of the "Final Resolution". If it fully addresses the customer's issue, conclude the process. Otherwise, return to "Inner Thinking" for further refinement.

</response requirements>

<customer ticket> represents the customer issue to be resolved, and <previous reasoning> contains your prior reasoning. Your task is to continue from the current 'Verification' step. Now, I'll secretly tell you that the correct resolution is "{}", but you must pretend not to know. Your 'Verification' requires careful consideration, and if incorrect, you need to provide new Inner Thinking steps and a new Final Resolution to ensure the final resolution aligns with the correct one.

### Output Format
Strictly follow the JSON structure below. You do not need to repeat your previous reasoning. Begin directly from the next 'Verification' stage.

```json
{{
"CoT": [
    {{"action": "Verification", "content": "..."}},
    {{"action": "Inner Thinking", "title": "...", "content": "..."}},
    ...,
    {{"action": "Final Resolution", "content": "..."}},
    {{"action": "Verification", "content": "..."}}
]
}}
```"""

reformat_to_complex_cot_prompt = """<Thought Process>
{}
</Thought Process>

<Customer Ticket>
{}
</Customer Ticket>

The <Thought Process> above reflects the model's reasoning based on the <Customer Ticket>. Your task is to rewrite the <Thought Process> to resemble a more human-like, intuitive natural thinking process. The new version should:

1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for transitions or validations, such as "hmm," "oh," "also," or "wait."
3. Expand the content, making the reasoning richer, more detailed, and logically clear while still being conversational and intuitive.

Return directly the revised natural thinking in JSON format as follows:
```json
{{
  "NaturalReasoning": "..."
}}
```"""

get_final_response_prompt = """<Internal Thinking>
{}
</Internal Thinking>

<Customer Ticket>
{}
</Customer Ticket>

The <Internal Thinking> represents your internal thoughts about the customer issue. Based on this, generate a rich and high-quality final response to the customer. Start with acknowledging their issue, then provide a clear resolution. Ensure your final response closely addresses all aspects of the customer's ticket. The response should be professional, empathetic, and helpful. Output only your final response, without any additional content."""

# search strategies
search_strategies = [('Backtracking',gen_prompt_rethink_Backtracking),('Exploring New Paths',gen_prompt_rethink_Exploring_New_Path),('Verification',gen_prompt_rethink_Verification),('Correction',gen_prompt_rethink_Correction)]



