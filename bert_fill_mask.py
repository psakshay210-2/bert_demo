import logging

# Suppress warnings to keep the output clean for the user
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import pipeline

def main():
    print("Loading BERT-Large model... (This is a large model [~1.3GB], please be patient!)")
    
    # Initialize the pipeline for Masked Language Modeling (filling in the blanks)
    # We use 'bert-large-uncased-whole-word-masking', a much stronger model than Base or DistilBERT.
    try:
        fill_mask = pipeline("fill-mask", model="bert-large-uncased-whole-word-masking")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*50)
    print("       BERT: FILL-IN-THE-BLANK DEMO")
    print("="*50)
    print("\nThis tool demonstrates how BERT 'reads' a sentence and understands context.")
    print("Instructions:")
    print("1. Type a sentence with the token '[MASK]' where you want BERT to guess the word.")
    print("2. Example: 'The doctor ran to the [MASK].'")
    print("3. Type 'exit' or 'quit' to stop the program.\n")

    while True:
        try:
            # Get input from the user
            text = input("Enter your sentence: ")
        except EOFError:
            break
        
        # Check for exit condition
        if text.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Validate input
        if '[MASK]' not in text:
            print("Context Missing: Your sentence must contain the token '[MASK]'. Please try again.\n")
            continue
            
        try:
            # Get predictions from BERT
            results = fill_mask(text)
            
            # If there's only one result, transformers might not return a list, so wrap it
            if not isinstance(results, list):
                results = [results]

            print("\nBERT predicts the following words for [MASK]:")
            print("-" * 40)
            print(f"{'WORD':<15} | {'CONFIDENCE':<10}")
            print("-" * 40)
            
            for result in results:
                word = result['token_str']
                score = result['score'] * 100 # Convert to percentage
                print(f"{word:<15} | {score:.2f}%")
            print("-" * 40 + "\n")
            
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main()
