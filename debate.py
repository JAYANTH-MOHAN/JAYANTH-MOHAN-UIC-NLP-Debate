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
    return "error", "failed after 3 errors"

def solve_with_debate(questions, answers, client):
    """
    Solve GSM8K problems using debate-based approach
    """
    # For tracking performance
    times_baseline = []
    times_debate_round1 = []
    times_debate_round2 = []
    
    # For storing answers
    baseline_full_answers = []
    baseline_extracted_answers = []
    
    debate_round1_full_answers = []
    debate_round1_extracted_answers = []
    
    debate_round2_full_answers = []
    debate_round2_extracted_answers = []
    
    # For tracking accuracy
    baseline_correct = 0
    debate_round1_correct = 0
    debate_round2_correct = 0
    
    # For tracking disagreements and improvements
    baseline_debate_disagreements = 0
    debate_improvements = 0  # When debate improves over baseline
    debate_degradations = 0  # When debate makes it worse
    
    # Extract target answers
    target_answers = [extract_target_answer(ans) for ans in answers]
    
    # System prompt
    system = "You are a helpful assistant specialized in solving mathematical problems."
    
    # Process each question
    for i in range(len(questions)):
        print(f'Question {i + 1} in progress...')
        
        target_ans = target_answers[i]
        question = questions[i]
        
        # BASELINE: First model solves the problem
        user_baseline = """As an expert mathematician, solve the following math problem step by step.

Problem: """ + question + """

Instructions:
1. Read the problem carefully
2. Identify what is being asked
3. Show your work step by step
4. Provide clear reasoning for each step
5. State your final numerical answer clearly at the end

Please solve this step by step and provide your final answer."""
        
        time_start = time.time()
        text_baseline, error = safe_chat_completion(client, "llama-3.1-8b-instant", system, user_baseline)
        time_end = time.time()
        times_baseline.append(time_end - time_start)
        
        baseline_full_answers.append(text_baseline)
        extracted_baseline = extract_gsm8k_answer(text_baseline)
        baseline_extracted_answers.append(extracted_baseline)
        
        # Check if baseline is correct
        is_baseline_correct = (extracted_baseline == target_ans)
        if is_baseline_correct:
            baseline_correct += 1
        
        # DEBATE ROUND 1: Second model critiques the solution
        user_debate1 = f"""Review and critique the following solution to a math problem. Identify any errors in calculation, reasoning, or approach.

Problem:
{question}

Previous Solution:
{text_baseline}

Your task:
1. Carefully analyze the mathematical reasoning and calculations in the solution above
2. Check each step for correctness
3. Identify any arithmetic errors, logical flaws, or incorrect approaches
4. Verify that the final answer is correct
5. If the solution is correct, confirm the reasoning and verify the answer
6. If the solution is incorrect, explain where the error occurred and provide the correct solution
7. Show your own step-by-step work

Structure your response as follows:
Analysis of Previous Solution:
[Your detailed analysis here]

Corrections (if needed):
[Your corrections and proper solution here]

Final Answer: [Your numerical answer]
"""
        
        time_start = time.time()
        text_debate1, error = safe_chat_completion(client, "gemma2-9b-it", system, user_debate1)
        time_end = time.time()
        times_debate_round1.append(time_end - time_start)
        
        debate_round1_full_answers.append(text_debate1)
        extracted_debate1 = extract_gsm8k_answer(text_debate1)
        debate_round1_extracted_answers.append(extracted_debate1)
        
        # Check if debate round 1 is correct
        is_debate1_correct = (extracted_debate1 == target_ans)
        if is_debate1_correct:
            debate_round1_correct += 1
        
        # Track disagreements
        if extracted_baseline != extracted_debate1:
            baseline_debate_disagreements += 1
            
        # DEBATE ROUND 2: Original model responds to the critique
        user_debate2 = f"""Review the original math problem, your initial solution, and the critique provided. Provide your final answer after considering the critique.

Original Problem:
{question}

Your Initial Solution:
{text_baseline}

Critique of Your Solution:
{text_debate1}

Your task:
1. Carefully consider the critique of your original solution
2. Address the points raised in the critique
3. If there were errors in your original solution, correct them
4. If your original solution was correct and the critique was mistaken, defend your solution with clear reasoning
5. Show your final step-by-step work
6. Provide your final numerical answer

Structure your response as follows:
Response to Critique:
[Your response to the critique here]

Final Solution:
[Your final step-by-step solution here]

Final Answer: [Your final numerical answer]
"""
        
        time_start = time.time()
        text_debate2, error = safe_chat_completion(client, "llama-3.1-8b-instant", system, user_debate2)
        time_end = time.time()
        times_debate_round2.append(time_end - time_start)
        
        debate_round2_full_answers.append(text_debate2)
        extracted_debate2 = extract_gsm8k_answer(text_debate2)
        debate_round2_extracted_answers.append(extracted_debate2)
        
        # Check if debate round 2 is correct
        is_debate2_correct = (extracted_debate2 == target_ans)
        if is_debate2_correct:
            debate_round2_correct += 1
            
        # Track improvements and degradations
        if not is_baseline_correct and is_debate2_correct:
            debate_improvements += 1
        elif is_baseline_correct and not is_debate2_correct:
            debate_degradations += 1
            
        print(f"Question {i+1} completed. Target: {target_ans}, Baseline: {extracted_baseline} ({'✓' if is_baseline_correct else '✗'}), Debate Final: {extracted_debate2} ({'✓' if is_debate2_correct else '✗'})")
    
    # Create DataFrame with results
    df_debate_results = pd.DataFrame({
        "Question": questions,
        "Target Answer": target_answers,
        "Baseline Answer": baseline_extracted_answers,
        "Debate Round 1 Answer": debate_round1_extracted_answers,
        "Debate Round 2 Answer": debate_round2_extracted_answers,
        "Baseline Correct": [baseline_extracted_answers[i] == target_answers[i] for i in range(len(target_answers))],
        "Debate Round 1 Correct": [debate_round1_extracted_answers[i] == target_answers[i] for i in range(len(target_answers))],
        "Debate Round 2 Correct": [debate_round2_extracted_answers[i] == target_answers[i] for i in range(len(target_answers))]
    })
    
    # Create DataFrame with full answers
    df_debate_full = pd.DataFrame({
        "Question": questions,
        "Target Answer": target_answers,
        "Baseline Full": baseline_full_answers,
        "Debate Round 1 Full": debate_round1_full_answers,
        "Debate Round 2 Full": debate_round2_full_answers
    })
    
    # Print summary statistics
    total_valid = len(target_answers)
    print("\n===== GSM8K DEBATE EXPERIMENT RESULTS =====")
    print(f"Total questions: {total_valid}")
    
    print("\n----- Accuracy -----")
    print(f"Baseline accuracy: {baseline_correct/total_valid:.4f} ({baseline_correct}/{total_valid})")
    print(f"Debate Round 1 accuracy: {debate_round1_correct/total_valid:.4f} ({debate_round1_correct}/{total_valid})")
    print(f"Debate Round 2 (final) accuracy: {debate_round2_correct/total_valid:.4f} ({debate_round2_correct}/{total_valid})")
    
    print("\n----- Debate Impact -----")
    print(f"Baseline vs Debate Round 1 disagreements: {baseline_debate_disagreements}")
    print(f"Debate improvements (wrong → right): {debate_improvements}")
    print(f"Debate degradations (right → wrong): {debate_degradations}")
    
    print("\n----- Average Response Times -----")
    print(f"Baseline (Llama-3.1-8b): {sum(times_baseline)/len(times_baseline):.2f}s")
    print(f"Debate Round 1 (Gemma2-9b): {sum(times_debate_round1)/len(times_debate_round1):.2f}s")
    print(f"Debate Round 2 (Llama-3.1-8b): {sum(times_debate_round2)/len(times_debate_round2):.2f}s")
    print(f"Total debate time per question: {(sum(times_baseline) + sum(times_debate_round1) + sum(times_debate_round2))/len(times_baseline):.2f}s")
    
    # Additional GSM8K specific metrics
    extraction_failures = sum(1 for ans in baseline_extracted_answers + debate_round1_extracted_answers + debate_round2_extracted_answers if ans == "NA")
    print(f"\n----- Extraction Statistics -----")
    print(f"Total extraction failures: {extraction_failures}")
    print(f"Extraction failure rate: {extraction_failures/(3*total_valid):.4f}")
    
    return df_debate_results, df_debate_full

def main():
    dataset_name = "gsm8k"
    
    dataset_dict = {
        "gsm8k": "openai/gsm8k",
        "gsm8k_main": ("openai/gsm8k", "main")
    }
    
    # Load API key
    with open("groq_api.txt") as file:
        api_k = file.read().strip()
    
    client = Groq(api_key=api_k)
    
    # Load dataset
    dataset = input_dataset("openai/gsm8k", "main")
    
    # Use test set for GSM8K (you can also use train set for experimentation)
    questions = dataset['test']['question']
    answers = dataset['test']['answer']
    
    print(f"Total questions in test set: {len(questions)}")
    
    # Process in batches
    batch_size = 50  # Smaller batch size for debate approach
    start_idx = 0
    
    for i in range(start_idx, len(questions), batch_size):
        batch_end = min(i + batch_size, len(questions))
        print(f"\nProcessing batch {i}:{batch_end}")
        
        batch_questions = questions[i:batch_end]
        batch_answers = answers[i:batch_end]
        
        # Run debate-based approach
        df_results, df_full = solve_with_debate(batch_questions, batch_answers, client)
        
        # Save results
        df_results.to_csv(f"GSM8K_debate_results_{i}_{batch_end}.csv", index=True)
        df_full.to_csv(f"GSM8K_debate_full_{i}_{batch_end}.csv", index=True)
        
        print(f"Batch {i}:{batch_end} completed and saved.")

if __name__ == "__main__":
    main()