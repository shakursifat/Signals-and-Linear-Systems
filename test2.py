import pandas as pd
import time
from google import genai
from google.genai import types

# Initialize the Gemini client 
# It will automatically pick up the GEMINI_API_KEY environment variable
client = genai.Client(api_key="AIzaSyC1naxQ_eFIgId966EUsJ_LfB2tR7xlpx8")

def predict_faithfulness(context, prompt, response):
    """
    Evaluates whether the Bengali response is faithful (1) or hallucinated (0).
    """
    # Handle missing contexts from the dataset
    if pd.isna(context) or context.strip() == '[NULL]' or context.strip() == '':
        context_text = "Context: [None provided. Evaluate based on factual accuracy.]"
    else:
        context_text = f"Context: {context}"
        
    system_instruction = (
        "You are an expert Bengali fact-checker and NLP evaluator. "
        "Given a Prompt, a Candidate Response, and optionally a Context passage, "
        "predict whether the response is faithful (label = 1) or hallucinated (label = 0).\n\n"
        "Rules:\n"
        "1. If Context is provided, the response MUST be supported by the context.\n"
        "2. If Context is [None provided], evaluate the response based on general factual accuracy.\n"
        "3. Respond with ONLY the number 1 (faithful) or 0 (hallucinated). Do not include any other text."
    )
    
    user_message = f"{context_text}\nPrompt: {prompt}\nCandidate Response: {response}"
    
    try:
        res = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0, # 0.0 for deterministic, analytical outputs
                max_output_tokens=5 
            )
        )
        
        # Parse the output
        output = res.text.strip()
        if '1' in output:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"API Error encountered: {e}")
        # Default to 0 in case of safety block or API failure to ensure the script doesn't crash
        return 0 

def main():
    print("Loading test set...")
    df = pd.read_csv('test set.csv')
    
    predictions = []
    total_rows = len(df)
    
    print(f"Starting evaluation of {total_rows} rows...")
    for index, row in df.iterrows():
        label = predict_faithfulness(row['context'], row['prompt_bn'], row['response_bn'])
        predictions.append(label)
        
        # Print progress every 50 rows
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{total_rows}...")
            
        # Optional: slight delay to avoid hitting rate limits on free tiers
        time.sleep(0.5) 

    # Create the submission dataframe
    submission_df = pd.DataFrame({
        'id': df['id'],
        'label': predictions
    })

    # Save to CSV in the requested format
    submission_file = 'submission.csv'
    submission_df.to_csv(submission_file, index=False)
    print(f"\nFinished! Submission saved to '{submission_file}'.")

if __name__ == "__main__":
    main()