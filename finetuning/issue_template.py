import random

# Sample input data (representing the provided 20,000 issues dataset)
issues_data = [{"issue": f"Issue {i}", "stepbystepaction": f"Resolution steps for issue {i}"} for i in range(1, 20001)]

# Given troubleshooting templates
templates = [
    "I need help with this issue: <<issue>>. What should I do?",
    "I’m encountering a problem: <<issue>>. Can you guide me through the resolution?",
    "I encountered the following issue: <<issue>>. What are the best troubleshooting steps?",
    "I need a fix to issue: <<issue>>. Can you provide a resolution?",
    "How do I fix this problem: <<issue>>?",
    "What’s the best way to resolve <<issue>>?",
    "What should I check first when troubleshooting <<issue>>?",
    "If I encounter problem: <<issue>>, what steps should I take to resolve it?",
    "What’s the recommended approach to solving problem: <<issue>>?",
    "Provide a step-by-step guide to resolving: <<issue>>.",
    "List the steps needed to diagnose and fix: <<issue>>.",
    "Explain how to troubleshoot and resolve: <<issue>> in detail.",
    "Break down the resolution process for issue: <<issue>> into actionable steps.",
    "Give me a structured plan to fix issue: <<issue>>.",
    "I need an immediate fix for this problem: <<issue>>. What’s the fastest way to resolve it?",
    "Critical issue: <<issue>> is disrupting operations. What should I do right now?",
    "<<issue>> I need to fix it ASAP. Guide me."
]

# Select 75% of issues randomly
selected_issues = random.sample(issues_data, int(0.75 * len(issues_data)))

# Generate formatted queries
formatted_data = [
    {
        "issue": issue["issue"],
        "stepbystepaction": issue["stepbystepaction"],
        "formatted_query": random.choice(templates).replace("<<issue>>", issue["issue"])
    }
    for issue in selected_issues
]

# Print a sample of the processed data
for entry in formatted_data[:5]:  # Display only first 5 for verification
    print(entry)
