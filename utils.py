import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_responses(text):
    """
    Extracts and returns the responses from the text, excluding the parts
    between and including the [INST] tags.

    Args:
    text (str): The input text containing responses and [INST] tags.

    Returns:
    str: The extracted responses.
    """
    import re

    # Split the text by [INST] tags and accumulate non-tag parts
    parts = re.split(r'\[INST\].*?\[/INST\]', text, flags=re.DOTALL)
    cleaned_text = "".join(parts)

    # Return the cleaned and trimmed text
    return cleaned_text.strip()


def generate_html():
  
  return(
      '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your Gradio App</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400&display=swap');

            body, html {
                margin: 0;
                padding: 0;
                font-family: 'Montserrat', sans-serif;
                background: #f9f9f9;
            }

            header {
                background-color: #e8f0fe;
                color: #333;
                text-align: center;
                padding: 40px 20px;
                border-radius: 0 0 25px 25px;
                background-image: linear-gradient(to right, #a7c7e7, #c0d8f0);
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
                position: relative;
                overflow: hidden;
            }

            .background-shapes {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: linear-gradient(120deg, #a7c7e7 0%, #c0d8f0 100%);
                opacity: 0.6;
                animation: pulse 5s ease-in-out infinite alternate;
            }

            .header-content h1 {
                font-size: 2.8em;
                margin: 0;
            }

            .header-content p {
                font-size: 1.3em;
                margin-top: 20px;
            }

            @keyframes pulse {
                from { background-size: 100% 100%; }
                to { background-size: 110% 110%; }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="background-shapes"></div>
            <div class="header-content">
                <h1>AI Assistant</h1>
                <p>This interactive app leverages the power of a fine-tuned Phi 2 AI model to provide insightful responses. Type your query below and witness AI in action.</p>
            </div>
        </header>
        <!-- Rest of your Gradio app goes here -->
    </body>
    </html>

  ''')

def generate_footer():
  
  return(
      '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your Gradio App</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

            body, html {
                margin: 0;
                padding: 0;
                font-family: 'Roboto Slab', serif;
                background: #f9f9f9;
            }

            header, footer {
                color: #333;
                text-align: center;
                padding: 40px 20px;
                border-radius: 25px;
                background: linear-gradient(120deg, #a7c7e7 0%, #c0d8f0 100%);
                background-size: 200% 200%;
                animation: gradientShift 8s ease-in-out infinite;
                position: relative;
                overflow: hidden;
            }

            .header-content, .footer-content {
                position: relative;
                z-index: 1;
            }

            .header-content h1, .footer-content p {
                font-size: 2.8em;
                margin: 0;
            }

            .header-content p, .footer-content p {
                font-size: 1.3em;
                margin-top: 20px;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            footer {
                margin-top: 40px;
                border-radius: 25px 25px 0 0;
            }
        </style>
    </head>
    <body>

        <footer>
            <div class="footer-content">
                <p>This model was fine-tuned on a subset of the OpenAssistant dataset.</p>
            </div>
        </footer>
    </body>
    </html>

  ''')



model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32, 
    device_map="cpu",
    trust_remote_code=True
)
model.load_adapter('checkpoint-780')


tokenizer = AutoTokenizer.from_pretrained('checkpoint-780', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token