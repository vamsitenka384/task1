# Text Summarization Tool
# By CodTech Internship Project

from transformers import pipeline

# Load summarizer only once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the input text using Hugging Face Transformers pipeline.
    
    Args:
        text (str): Input text/article.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: Concise summary of the text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":
    # Example input article
    article = """
    Artificial Intelligence (AI) is one of the most transformative technologies of the 21st century. 
    It has applications across multiple domains including healthcare, finance, transportation, and education. 
    AI enables machines to mimic human intelligence, learning from data and improving over time. 
    Companies worldwide are investing heavily in AI research and development. 
    Despite its potential, AI also raises ethical concerns such as job displacement, privacy issues, 
    and the possibility of biased decision-making. Governments and organizations are now working to 
    create regulations that ensure responsible use of AI while maximizing its benefits for society.
    """

    print("Original Article:\n", article)
    print("\n--- Summary ---")
    print(summarize_text(article))
