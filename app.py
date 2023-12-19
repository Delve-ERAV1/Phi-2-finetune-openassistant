import gradio as gr
import random
import torch
from utils import *


with gr.Blocks() as demo:

  with gr.Row():
    show_label = True
    gr.HTML(value=generate_html, show_label=show_label)

  with gr.Row():
    temp = gr.Slider(0, 1, value=0.2, label="Temperature", info="Choose between 0 and 1")
    seed = gr.Slider(0, 1000, value=42, label="Seed", info="Select Random Seed")
    max_tokens = gr.Slider(100, 1000, value=200, label="Max Tokens", info="Choose Max Tokens")

  with gr.Row():
    with gr.Column():
      chatbot = gr.Chatbot()
      msg = gr.Textbox(label='Message AI Assistant')
      clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history, temp, seed, max_tokens):

        torch.manual_seed(seed)
        model_inputs = tokenizer(
                    [f"[INST] {message} [/INST]"], 
                    return_tensors="pt", padding=True)
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens)
        result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        bot_message = extract_responses(result[0])
        chat_history.append((message, bot_message))
    
        return "", chat_history

    msg.submit(respond, [msg, chatbot, temp, seed, max_tokens], [msg, chatbot])

  with gr.Row():
    show_label = True
    gr.HTML(value=generate_footer, show_label=show_label)

if __name__ == "__main__":
    demo.launch()

