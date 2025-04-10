#!/usr/bin/env python3
"""
QA Pair Generation using Qwen2.5-1.5B-Instruct model.

This script loads alloy data from JSON, generates question-answer pairs,
filters them using heuristics and model evaluation, and saves high-quality pairs.
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model and tokenizer
MODEL = None
TOKENIZER = None

def initialize_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Initialize the model and tokenizer."""
    global MODEL, TOKENIZER
    print(f"Initializing model: {model_name}")
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model initialized successfully")

def load_context_from_file(file_path, limit=None):
    """Load context from JSON and extract relevant information."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        contexts = []
        context_counter = 0

        for entry in data:
            if "alloys" in entry and isinstance(entry["alloys"], list):
                alloy_contexts = {}

                # Iterate over each alloy within the same context
                for alloy_data in entry["alloys"]:
                    context = {
                        "name": alloy_data.get("name", ""),
                        "composition": alloy_data.get("composition", {}),
                        "performance_properties": alloy_data.get("performance_properties", {}),
                        "material_conditions": alloy_data.get("material_conditions", {}),
                        "doi": alloy_data.get("doi", "")
                    }

                    # Convert the context to JSON string for use as a key
                    context_str = json.dumps(context, ensure_ascii=False, indent=2)
                    alloy_contexts[context_str] = context_str

                # Store context with multiple alloys
                contexts.append(alloy_contexts)
                context_counter += 1

                # Stop if limit is reached
                if limit is not None and context_counter >= limit:
                    break

        return contexts

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading context: {e}")
        return []

def clean_and_parse_response(response):
    """Clean and parse model response by removing code block formatting."""
    if response.startswith("```json") and response.endswith("```"):
        response = response.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {"error": "Invalid JSON format", "response": response}

def generate_qa_for_context(context_str):
    """Generate QA pairs for a given alloy context and parse the output."""
    prompt = f""" 
    Generate 8 diverse questions for each category along with their answer pairs from the given context.
    Do not make assumptions or include unrelated information.
    Cover multiple categories such as:
    - Property Value-Based Questions: Retrieve exact numerical values of material mechanical properties from the given context.
    - Temperature Based Questions: Ask about the temperature under which the properties were measured.
    - Unit-Based Questions: Identify the units associated with the property values provided in the context.
    - Material Condition Based: Inquire about the equilibrium conditions, phase_type, and related conditions for the alloy.

    ### Format the output strictly as a valid JSON with the following structure:
    {{
        "Property Value-Based Questions": [
            {{"question": "<question_1 containing alloy_name in the context>", "answer": "<answer_1>"}},
            {{"question": "<question_2 containing alloy_name in the context>", "answer": "<answer_2>"}}
        ],
        "Temperature Based Questions": [
            {{"question": "<question_1 containing alloy_name in the context>", "answer": "<answer_1>"}}
        ],
        "Unit-Based Questions": [
            {{"question": "<question_1 containing alloy_name in the context>", "answer": "<answer_1>"}}
        ],
        "Material Condition Based": [
            {{"question": "<question_1 containing alloy_name in the context>", "answer": "<answer_1>"}}
        ]
    }}

    Context:
    {context_str}
    """

    # Define messages for the chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template and prepare input
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)

    # Optimized Generation Parameters
    generated_ids = MODEL.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.05,
    )

    # Remove input prompt from generated output
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated output
    response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean and parse the response immediately after generating it
    parsed_response = clean_and_parse_response(response)

    return parsed_response

def load_qa_pairs(file_path):
    """Load Q&A pairs from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading Q&A pairs: {e}")
        return {}

def validate_qa_data(qa_data):
    """Validate the format of loaded Q&A pairs."""
    if not isinstance(qa_data, dict):
        raise ValueError("❌ Invalid format: Expected a dictionary at the root level.")

    for context_key, context_results in qa_data.items():
        if not isinstance(context_results, dict):
            raise ValueError(f"❌ Invalid format under context '{context_key}'. Expected dictionary.")

        for alloy_name, qa_categories in context_results.items():
            if not isinstance(qa_categories, dict):
                raise ValueError(f"❌ Invalid format for alloy '{alloy_name}'. Expected category dictionary.")

            for category_name, qa_pairs in qa_categories.items():
                if not isinstance(qa_pairs, list):
                    raise ValueError(f"❌ Invalid format for category '{category_name}'. Expected list of QA pairs.")

                for qa in qa_pairs:
                    if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                        raise ValueError(f"❌ Malformed QA pair found: {qa}")

def heuristic_filter_and_clean_qa_pairs(input_file, output_file):
    """
    Remove Q-A pairs containing placeholders and clean extra keys.
    
    Args:
        input_file (str): Path to the input file with generated Q-A pairs.
        output_file (str): Path to save the cleaned Q-A pairs.
    """
    qa_data = load_qa_pairs(input_file)

    if not qa_data:
        raise ValueError("❌ No valid Q&A pairs found in the input file. Check the file format.")

    # Validate data structure before proceeding
    validate_qa_data(qa_data)

    cleaned_data = {}

    # Regex pattern to detect placeholders
    placeholder_pattern = re.compile(r"<\s*(?:Value|Unit)\s*>", re.IGNORECASE)

    # Iterate through all contexts and alloys
    for context_key, context_results in qa_data.items():
        cleaned_data[context_key] = {}

        for alloy_name, qa_categories in context_results.items():
            cleaned_data[context_key][alloy_name] = {}

            for category_name, qa_pairs in qa_categories.items():
                cleaned_qa_list = []

                # Check each Q-A pair for placeholders and clean extra keys
                for qa in qa_pairs:
                    # Remove any extra keys by keeping only 'question' and 'answer'
                    cleaned_qa = {
                        "question": qa["question"],
                        "answer": qa["answer"]
                    }

                    # Skip pairs with placeholder values like <Value> or <Unit>
                    if not placeholder_pattern.search(qa["answer"]):
                        cleaned_qa_list.append(cleaned_qa)
                    else:
                        print(f"🛑 Skipped Q-A pair with placeholders: [Alloy: {alloy_name}] - {qa['question']}")

                # Only retain non-empty results
                if cleaned_qa_list:
                    cleaned_data[context_key][alloy_name][category_name] = cleaned_qa_list

    # Save cleaned data to output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(cleaned_data, outfile, indent=4, ensure_ascii=False)

    print(f"\n✅ Cleaned and filtered Q&A pairs saved to {output_file}")

def evaluate_qa_pairs(context_str, alloy_name, qa_pairs):
    """Evaluate each Q&A pair and return a score between 1 and 10."""
    evaluated_pairs = {}

    # Updated Evaluation Prompt Template
    eval_prompt_template = """
    ### Evaluation Task: Evaluate the quality of Q-A pairs based on **category alignment** and **contextual accuracy**. Assign a score between **1 and 10**.

    ### Rating Scale:
    - **1-3:** Poor Q-A pair, irrelevant, incorrect, or major inaccuracies.
    - **4-6:** Fair Q-A pair, partially relevant with minor inconsistencies.
    - **7-8:** Good Q-A pair, mostly accurate and relevant with minor flaws.
    - **9-10:** Excellent Q-A pair, highly relevant, perfectly aligned, and accurate.

    ### Evaluation Criteria:
    1. **Category Validation:**
       - Check if the question matches the correct category:
         - **Property Value-Based:** Retrieve numerical values of material properties.
           - *Format:* "What is the [property] of [alloy] at [temperature]?"
         - **Temperature-Based:** Ask about the temperature at which a property is measured.
           - *Format:* "At what temperature is the [property] of [alloy] measured?"
         - **Unit-Based:** Identify the units for property values.
           - *Format:* "What is the unit of the [property] for [alloy]?"
         - **Material Condition-Based:** Inquire about equilibrium conditions or phase types.
           - *Format:* "What are the equilibrium conditions of [alloy]?"

    2. **Contextual Accuracy:**
       - Ensure the Q-A pair is consistent with the context.
       - Penalize mismatches and placeholder values (`<Value>`, `<Unit>`).

    3. **Answer Assessment:**
       - Check for correctness, completeness, and specificity.
       - Penalize misleading, incomplete, or irrelevant answers.

    ### Evaluation Workflow:
    1. *Review Context:* Analyze alloy details.
    2. *Verify Category:* Check correct categorization.
    3. *Assess Answer:* Evaluate relevance and accuracy.
    4. *Check for Placeholders:* Penalize incomplete placeholders.
    5. *Assign and Justify Score:* Provide a brief rationale.

    ### Important:
    - **Always provide the rating strictly in the format:** `### Rating: [1-10]`
    - **Failure to include a valid rating will result in the lowest score (1).**

    ### Submission Format:
    Rating: [1-10]
    Justification: Brief explanation
    Key Observations: Notable insights

    ### Context:
    {context_str}

    ### Alloy:
    {alloy_name}

    ### Question:
    {question}

    ### Answer:
    {answer}
    """

    for category, qa_list in qa_pairs.items():
        rated_qa_list = []
        print(f"➡️ Evaluating {len(qa_list)} Q&A pairs for category: {category}...")

        for idx, qa in enumerate(qa_list):
            # Create prompt for each Q&A pair
            eval_prompt = eval_prompt_template.format(
                context_str=context_str,
                alloy_name=alloy_name,
                question=qa["question"],
                answer=qa["answer"]
            )

            # Prepare input for the model
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": eval_prompt}
            ]
            text = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)

            # Generate the evaluation score
            generated_ids = MODEL.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            # Decode the evaluation response
            eval_response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract rating using regex and fallback to 1 if missing
            rating_match = re.search(r"### Rating:\s*(\d+)", eval_response)

            if rating_match and rating_match.group(1).isdigit():
                score = int(rating_match.group(1))
                if score < 1 or score > 10:
                    print(f"⚠️ Invalid score ({score}) detected. Defaulting to 1.")
                    score = 1
            else:
                print("⚠️ No valid rating found. Assigning default score: 1.")
                score = 1  # Fallback in case of no valid rating

            print(f"✅ Q{idx + 1}/{len(qa_list)} - Rated: {score}/10")

            # Append evaluated QA pair
            rated_qa_list.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "rating": score
            })

        evaluated_pairs[category] = rated_qa_list

    return evaluated_pairs

def main_evaluation(qa_input_file, eval_output_file):
    """Main function to evaluate generated QA pairs and save ratings."""
    # Load generated Q&A pairs
    qa_data = load_qa_pairs(qa_input_file)

    if not qa_data:
        raise ValueError("❌ No valid Q&A pairs found. Please check the file format and content.")

    eval_results = {}

    # Iterate over each context and evaluate
    for context_key, context_results in qa_data.items():
        print(f"\n📚 Processing Context: {context_key}")
        eval_results_per_context = {}

        for alloy_name, qa_pairs in context_results.items():
            # Evaluate QA pairs and store results
            evaluated_pairs = evaluate_qa_pairs(context_key, alloy_name, qa_pairs)
            eval_results_per_context[alloy_name] = evaluated_pairs

        eval_results[context_key] = eval_results_per_context

    # Save evaluated QA pairs with ratings
    with open(eval_output_file, "w", encoding="utf-8") as outfile:
        json.dump(eval_results, outfile, indent=4, ensure_ascii=False)

    print(f"\n✅ Evaluated QA pairs with ratings saved to {eval_output_file}")

def load_evaluated_qa_pairs(file_path):
    """Load evaluated Q&A pairs from the JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading evaluated Q&A pairs: {e}")
        return {}

def filter_qa_pairs(eval_data, rating_threshold=6):
    """Filter Q&A pairs with ratings > threshold and remove 'rating' key."""
    filtered_results = {}

    for context_key, context_results in eval_data.items():
        print(f"🔍 Processing {context_key}...")

        filtered_context_results = {}

        for alloy_name, qa_categories in context_results.items():
            filtered_qa_categories = {}

            for category, qa_list in qa_categories.items():
                # Filter pairs with rating > threshold or == 0
                filtered_qa_list = [
                    {k: v for k, v in qa.items() if k != "rating"}  # Remove 'rating' key
                    for qa in qa_list
                    if qa["rating"] > rating_threshold or qa["rating"] == 0
                ]

                if filtered_qa_list:
                    filtered_qa_categories[category] = filtered_qa_list

            if filtered_qa_categories:
                filtered_context_results[alloy_name] = filtered_qa_categories

        if filtered_context_results:
            filtered_results[context_key] = filtered_context_results

    return filtered_results

def save_filtered_qa_pairs(filtered_data, output_file):
    """Save filtered Q&A pairs to the specified file."""
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)

    print(f"✅ Filtered Q&A pairs saved to {output_file}")

def main_filtering(eval_input_file, filtered_output_file, rating_threshold=6):
    """Main function to load, filter, and save filtered QA pairs."""
    # Load evaluated Q&A pairs
    eval_data = load_evaluated_qa_pairs(eval_input_file)

    if not eval_data:
        raise ValueError("❌ No valid evaluated Q&A pairs found. Please check the input file format.")

    # Filter Q&A pairs and remove 'rating' key
    filtered_results = filter_qa_pairs(eval_data, rating_threshold)

    # Save filtered results
    save_filtered_qa_pairs(filtered_results, filtered_output_file)

def generate_qa_pairs(input_file, output_file, limit=None):
    """Main function to process the JSON and generate QA pairs."""
    # Load contexts dynamically
    contexts = load_context_from_file(input_file, limit)

    if not contexts:
        raise ValueError("No valid context found. Please check the file format and content.")

    combined_results = {}

    # Generate Q&A pairs for each context containing multiple alloys
    for idx, context in enumerate(contexts):
        print(f"Generating Q&A pairs for context {idx + 1}...")
        
        context_results = {}

        # Iterate over each context string (original alloy context as key)
        for context_str, original_context in context.items():
            response = generate_qa_for_context(context_str)

            # Store response with context string as the key
            if isinstance(response, dict) and "error" not in response:
                context_results[context_str] = response
            else:
                context_results[context_str] = {"error": "Invalid JSON format", "response": response}

        # Store all responses under context_{idx + 1}
        combined_results[f"context_{idx + 1}"] = context_results

    # Save combined results to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(combined_results, outfile, indent=4, ensure_ascii=False)

    print(f"QA pairs successfully generated and saved to {output_file}")
    
    return output_file

def main(input_json, output_dir="./", limit=None, rating_threshold=6):
    """
    Main pipeline for generating, filtering, evaluating, and saving QA pairs.
    
    Args:
        input_json (str): Path to input JSON file with alloy data
        output_dir (str): Directory to save output files
        limit (int, optional): Limit the number of contexts to process
        rating_threshold (int, optional): Minimum rating threshold for final filtering
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    generated_qa_file = os.path.join(output_dir, "generated_qa_pairs.json")
    filtered_qa_file = os.path.join(output_dir, "filtered_qa_pairs.json")
    eval_output_file = os.path.join(output_dir, "qa_pair_ratings.json")
    final_output_file = os.path.join(output_dir, "final_qa_pairs.json")
    
    # Initialize model
    initialize_model()
    
    # Step 1: Generate QA pairs
    print("\n=== STEP 1: GENERATING QA PAIRS ===")
    generate_qa_pairs(input_json, generated_qa_file, limit)
    
    # Step 2: Apply heuristic filtering
    print("\n=== STEP 2: APPLYING HEURISTIC FILTERING ===")
    heuristic_filter_and_clean_qa_pairs(generated_qa_file, filtered_qa_file)
    
    # Step 3: Evaluate QA pairs
    print("\n=== STEP 3: EVALUATING QA PAIRS ===")
    main_evaluation(filtered_qa_file, eval_output_file)
    
    # Step 4: Final filtering based on ratings
    print("\n=== STEP 4: FINAL FILTERING ===")
    main_filtering(eval_output_file, final_output_file, rating_threshold)
    
    print(f"\n✅ QA pair generation pipeline completed successfully!")
    print(f"Final QA pairs available at: {final_output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QA pairs from alloy data")
    parser.add_argument("input_json", help="Path to input JSON file containing alloy data")
    parser.add_argument("--output_dir", default="./", help="Directory to save output files")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of contexts to process")
    parser.add_argument("--rating_threshold", type=int, default=6, help="Minimum rating threshold for final filtering")
    
    args = parser.parse_args()
    
    main(args.input_json, args.output_dir, args.limit, args.rating_threshold)

