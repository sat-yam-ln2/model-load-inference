import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import json

def get_available_models():
    """Get a list of available models from the cache directory."""
    model_dir = "cache/model"
    if os.path.exists(model_dir):
        return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    return []

def load_model_and_tokenizer(model_name, use_gpu):
    """Load the tokenizer and model for the given model name, with an option to use GPU."""
    try:
        tokenizer_path = f"cache/tokenizer/{model_name}"
        model_path = f"cache/model/{model_name}"
        
        # Check if this is a Qwen model and update config if necessary
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'Qwen' in model_name and 'model_type' not in config:
                    config['model_type'] = 'qwen' if 'Qwen2' not in model_name else 'qwen2'
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
        
        # Load tokenizer with trust_remote_code=True for Qwen models
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True if 'Qwen' in model_name else False
        )
        
        # Load model with trust_remote_code=True for Qwen models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True if 'Qwen' in model_name else False,
            device_map='auto' if use_gpu and torch.cuda.is_available() else None
        )
        
    except Exception as e:
        raise ValueError(f"Error loading model or tokenizer: {str(e)}")

    return tokenizer, model

def generate_response(model_name, input_text, use_gpu, max_length, num_return_sequences):
    """Generate a response using the selected model and tokenizer with specified parameters."""
    tokenizer, model = load_model_and_tokenizer(model_name, use_gpu)
    
    # Special handling for Qwen models
    if 'Qwen' in model_name:
        inputs = tokenizer(input_text, return_tensors="pt")
        if use_gpu and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            trust_remote_code=True
        )
    else:
        inputs = tokenizer(input_text, return_tensors="pt")
        if use_gpu and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return "\n\n".join(responses)

def run_app():
    """Launch the Gradio app."""
    available_models = get_available_models()
    if not available_models:
        raise FileNotFoundError("No models found in the cache directory. Please add models to 'cache/model/'.")

    with gr.Blocks() as demo:
        gr.Markdown("""
        # Transformer Model Selector
        Choose a model, input text, and get generated output!
        """)

        model_selector = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="Select a Model"
        )

        use_gpu = gr.Checkbox(label="Use GPU (if available)", value=False)
        input_text = gr.Textbox(label="Input Text")
        max_length = gr.Slider(label="Max Length", minimum=10, maximum=200, step=10, value=100)
        num_return_sequences = gr.Slider(label="Number of Return Sequences", minimum=1, maximum=5, step=1, value=1)
        output_text = gr.Textbox(label="Generated Output", interactive=False)

        generate_button = gr.Button("Generate")

        generate_button.click(
            fn=generate_response,
            inputs=[model_selector, input_text, use_gpu, max_length, num_return_sequences],
            outputs=output_text
        )

    demo.launch(share=True)

if __name__ == "__main__":
    run_app()