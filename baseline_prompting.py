from datasets import load_dataset 
import pandas as pd
from groq import Groq
import time
import re

def input_dataset(data_name, version=""):
    if version == "":
        dataset = load_dataset(data_name, trust_remote_code=True)
    else:
        dataset = load_dataset(data_name, version, trust_remote_code=True)
    return dataset

def convert_to_number(s):
    """Convert string to number, handling various formats"""
    s = s.replace(' ', '').replace("'", '').replace('"', '').replace(',', '')
    if s == "NA":
        return s
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return "NA"

def extract_gsm8k_answer(text):
    """
    Extract the final numerical answer from GSM8K response.
    Looks for patterns like final answer numbers or numbers at the end
    """
    # Clean the text
    text = text.strip()
    
    # Look for "Final Answer:" pattern followed by a number
    final_answer_match = re.search(r"Final Answer:\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if final_answer_match:
        return convert_to_number(final_answer_match.group(1))
    
    # Look for "Answer:" pattern followed by a number
    answer_match = re.search(r"Answer:\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if answer_match:
        return convert_to_number(answer_match.group(1))
    
    # Look for numbers in parentheses or after equals sign
    parentheses_match = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", text)
    if parentheses_match:
        return convert_to_number(parentheses_match.group(1))
    
    equals_match = re.search(r"=\s*([+-]?\d+(?:\.\d+)?)", text)
    if equals_match:
        return convert_to_number(equals_match.group(1))
    
    # Fallback: get the last number in the text
    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)', text)
    if numbers:
        return convert_to_number(numbers[-1])
    
    return "NA"

def extract_target_answer(gsm_answer):
    """Extract the target answer from GSM8K answer format"""
    # GSM8K answers typically end with "#### NUMBER"
    match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', gsm_answer)
    if match:
        return convert_to_number(match.group(1))
    
    # Fallback: look for last number
    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)', gsm_answer)
    if numbers:
        return convert_to_number(numbers[-1])
    
    return "NA"

def safe_chat_completion(client, model, system_prompt, user_prompt, temperature=0.0, max_tokens=1000, retries=3, backoff=2):
    """Safe API call with retry logic"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content, None
        except Exception as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                print(f"[Attempt {attempt+1}] 503 Service Unavailable — retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
            else:
                return "error", str(e)
    return "error", "failed after 3 retries"

def test_single_model(questions, answers, client, model_name="llama-3.1-8b-instant", custom_prompt=None):
    """
    Test a single model on GSM8K problems with a custom prompt
    """
    # For tracking performance
    response_times = []
    
    # For storing answers
    full_answers = []
    extracted_answers = []
    
    # For tracking accuracy
    correct_count = 0
    
    # Extract target answers
    target_answers = [extract_target_answer(ans) for ans in answers]
    
    # Default system prompt
    system_prompt = "You are a helpful assistant specialized in solving mathematical problems step by step."
    
    # Default user prompt template (if no custom prompt provided)
    default_prompt_template = """Solve the following math problem step by step.

Problem: {question}

Instructions:
1. Read the problem carefully
2. Identify what is being asked
3. Show your work step by step with clear reasoning
4. Perform calculations accurately
5. State your final numerical answer clearly

Please solve this step by step and provide your Final Answer: [number]"""
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt is None:
        prompt_template = default_prompt_template
    else:
        prompt_template = custom_prompt
    
    # Process each question
    for i in range(len(questions)):
        print(f'Question {i + 1}/{len(questions)} in progress...')
        
        target_ans = target_answers[i]
        question = questions[i]
        
        # Format the prompt with the question
        if "{question}" in prompt_template:
            user_prompt = prompt_template.format(question=question)
        else:
            # If no placeholder, append question to the prompt
            user_prompt = prompt_template + "\n\nProblem: " + question
        
        # Get model response
        time_start = time.time()
        text_response, error = safe_chat_completion(client, model_name, system_prompt, user_prompt)
        time_end = time.time()
        response_times.append(time_end - time_start)
        
        if error:
            print(f"Error on question {i+1}: {error}")
            full_answers.append(f"ERROR: {error}")
            extracted_answers.append("NA")
        else:
            full_answers.append(text_response)
            extracted_answer = extract_gsm8k_answer(text_response)
            extracted_answers.append(extracted_answer)
            
            # Check if answer is correct
            is_correct = (extracted_answer == target_ans)
            if is_correct:
                correct_count += 1
            
            print(f"Question {i+1}: Target={target_ans}, Model={extracted_answer} ({'✓' if is_correct else '✗'})")
    
    # Create DataFrame with results
    df_results = pd.DataFrame({
        "Question_ID": range(1, len(questions) + 1),
        "Question": questions,
        "Target_Answer": target_answers,
        "Model_Answer": extracted_answers,
        "Correct": [extracted_answers[i] == target_answers[i] for i in range(len(target_answers))],
        "Response_Time": response_times
    })
    
    # Create DataFrame with full responses
    df_full = pd.DataFrame({
        "Question_ID": range(1, len(questions) + 1),
        "Question": questions,
        "Target_Answer": target_answers,
        "Full_Response": full_answers,
        "Extracted_Answer": extracted_answers
    })
    
    # Print summary statistics
    total_questions = len(questions)
    accuracy = correct_count / total_questions
    avg_response_time = sum(response_times) / len(response_times)
    
    print("\n" + "="*50)
    print("GSM8K SINGLE MODEL TEST RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average response time: {avg_response_time:.2f}s")
    print(f"Total time: {sum(response_times):.2f}s")
    
    # Extraction statistics
    extraction_failures = sum(1 for ans in extracted_answers if ans == "NA")
    print(f"Extraction failures: {extraction_failures}")
    print(f"Extraction failure rate: {extraction_failures/total_questions:.4f}")
    
    return df_results, df_full

def main():
    # Configuration
    MODEL_NAME = "llama-3.1-8b-instant"  # Change this to test different models
    BATCH_SIZE = 10  # Number of questions to process
    START_IDX = 0    # Starting index (useful for continuing from where you left off)
    
    # Custom prompt (optional) - modify this to test different prompting strategies
    CUSTOM_PROMPT = """You are an expert mathematician. Solve the following math problem with careful step-by-step reasoning.

Problem: {question}

Approach:
1. Understand what the problem is asking
2. Break down the problem into smaller steps
3. Show all calculations clearly
4. Double-check your arithmetic
5. Provide your final answer as a number

Work through this systematically and end with: Final Answer: [your numerical answer]"""
    
    # Set to None to use default prompt
    # CUSTOM_PROMPT = None
    
    # Load API key
    try:
        with open("groq_api.txt") as file:
            api_key = file.read().strip()
    except FileNotFoundError:
        print("Error: groq_api.txt file not found. Please create it with your API key.")
        return
    
    client = Groq(api_key=api_key)
    
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = input_dataset("openai/gsm8k", "main")
    
    # Use test set
    questions = dataset['test']['question']
    answers = dataset['test']['answer']
    
    print(f"Total questions available: {len(questions)}")
    
    # Process subset
    end_idx = min(START_IDX + BATCH_SIZE, len(questions))
    batch_questions = questions[START_IDX:end_idx]
    batch_answers = answers[START_IDX:end_idx]
    
    print(f"Processing questions {START_IDX+1} to {end_idx}")
    print(f"Model: {MODEL_NAME}")
    print(f"Custom prompt: {'Yes' if CUSTOM_PROMPT else 'No (using default)'}")
    
    # Run the test
    df_results, df_full = test_single_model(
        batch_questions, 
        batch_answers, 
        client, 
        model_name=MODEL_NAME,
        custom_prompt=CUSTOM_PROMPT
    )
    
    # Save results
    results_filename = f"GSM8K_single_model_{MODEL_NAME.replace('/', '_')}_{START_IDX}_{end_idx}_results.csv"
    full_filename = f"GSM8K_single_model_{MODEL_NAME.replace('/', '_')}_{START_IDX}_{end_idx}_full.csv"
    
    df_results.to_csv(results_filename, index=False)
    df_full.to_csv(full_filename, index=False)
    
    print(f"\nResults saved to: {results_filename}")
    print(f"Full responses saved to: {full_filename}")

if __name__ == "__main__":
    main()