# The challenge with streaming in FastAPI is that the model generation is a synchronous, blocking process. If we run it directly in the main thread, the API can't do anything else (like handle other requests or health checks) while it's "thinking." To solve this, we can run the generation in a separate thread. This way, the main thread can immediately start yielding tokens to the client as they become available, without waiting for the entire generation to finish.
import os
import torch
from pathlib import Path
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer
from IPython.display import display, Markdown, HTML
from unsloth import FastLanguageModel # Assuming Unsloth based on your folder structure

# Configuration
MODEL_NAME = "unsloth/Qwen3.5-2B"
ADAPTER_MODEL_PATH = "./outputs/checkpoint-120" # Path to your fine-tuned adapter

MAX_NEW_TOKENS = 1024

class InferenceEngine:
    def __init__(self, model_name=MODEL_NAME, adapter_path=ADAPTER_MODEL_PATH):
        # 1. Load the model and tokenizer
        # We use Unsloth's FastLanguageModel for 2x faster inference
        self.model, self.processor = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_16bit=True,
        )
        # 2. CRITICAL: Ensure model is fully on GPU before loading adapter
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)  # Force to GPU
        self.tokenizer = self.processor.tokenizer  # Get the tokenizer from the model's processor
        # 1. Load base model (same as training)
        # 2. Load your fine-tuned adapter
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = Path(__file__).parent.parent.parent.resolve() # Get the directory where this file sits
        # Use the / operator to join paths (it's smart!)
        # .resolve() ensures all ./ and ../ are removed
        actual_adapter_path = (script_dir / adapter_path).resolve()
        # actual_adapter_path = os.path.join(script_dir, adapter_path)
        self.model.load_adapter(actual_adapter_path)
        # 3. Enable fast inference (2x speedup)
        FastLanguageModel.for_inference(self.model)
        # Verify all on same device
        print(f"Model device: {next(self.model.parameters()).device}")  # should say cuda:0
        self.model.to(device)  # Ensure model is on GPU after loading adapter


    def generate_stream(self, prompt: str):
        """
        A generator function that yields tokens one by one.
        """
        # 2. Prepare inputs
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # 3. Setup the streamer
        # timeout=10 ensures we don't hang forever if the thread dies
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # 4. Define generation arguments
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 5. Start generation in a separate thread
        # This allows the main thread to immediately start reading from the streamer
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        # 3. Create a Smooth Display Handle
        # We initialize it with a 'thinking' message or empty string
        display_handle = display(Markdown("Thinking..."), display_id=True)
        full_text = ""
   

        # 6. Yield tokens as they become available
        for new_text in streamer:
            full_text += new_text
            # Optional: Add a subtle 'cursor' like ChatGPT
            # streaming_html = f"{full_text}█"
            # Update the existing handle without clearing the whole output
            # display_handle.update(Markdown(streaming_html))

            # # 5. Final Update to remove the cursor once finished
            # display_handle.update(Markdown(full_text))
            yield new_text  # Yield the new token to the client immediately
        if display_handle:
            display_handle.update(Markdown(full_text))  # Final update to remove cursor
        thread.join()

# Global instance so the model only loads ONCE when the API starts
engine = InferenceEngine()