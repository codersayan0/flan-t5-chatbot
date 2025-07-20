import gradio as gr
from transformers import pipeline

# Load model
chatbot = pipeline("text2text-generation", model="google/flan-t5-base")

# Store chat history globally
chat_log = []

# Generate response and update history
def generate_response(prompt):
    full_prompt = f"Answer this clearly: {prompt}"
    result = chatbot(full_prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]

    # Update chat log
    chat_log.append(f"You: {prompt}")
    chat_log.append(f"Bot: {result}")

    # Join chat log for display
    return "\n".join(chat_log)

# Export chat history to text file
def export_chat():
    filename = "chat_history.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(chat_log))
    return filename

# Interface layout
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 FLAN-T5 Chatbot\nAsk anything, get clear answers. You can also download your conversation.")
    
    chatbox = gr.Textbox(label="Chat", lines=10, interactive=False)

    with gr.Row():
        input_box = gr.Textbox(placeholder="Type your question...", label="Your Message")
        submit_btn = gr.Button("Send")
        download_btn = gr.Button("Download Chat")

    # Update chat
    def handle_chat(prompt):
        response = generate_response(prompt)
        return response

    submit_btn.click(fn=handle_chat, inputs=input_box, outputs=chatbox)
    download_btn.click(fn=export_chat, inputs=[], outputs=gr.File(label="Your Download"))

demo.launch()
