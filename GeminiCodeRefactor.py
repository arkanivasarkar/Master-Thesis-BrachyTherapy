import sys
import os
import google.generativeai as genai

# Configure Gemini with your API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-pro")

def improve_code(code_snippet: str) -> str:
    """
    Takes a code snippet, asks Gemini to add comments and optimize it.
    Returns improved code.
    """
    prompt = f"""
    You are a Python expert. Improve the following code by:
    1. Adding clear, concise comments.
    2. Optimizing performance or readability where possible.
    3. Returning only the improved code.

    Code:
    {code_snippet}
    """

    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python refactor.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    # Read file contents
    with open(filename, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Get improved code
    improved = improve_code(original_code)

    # Overwrite file with improved code
    with open(filename, "w", encoding="utf-8") as f:
        f.write(improved)

    print(f"✅ Refactored {filename} successfully.")
