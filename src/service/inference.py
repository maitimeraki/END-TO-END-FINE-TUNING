# The challenge with streaming in FastAPI is that the model generation is a synchronous, blocking process. If we run it directly in the main thread, the API can't do anything else (like handle other requests or health checks) while it's "thinking." To solve this, we can run the generation in a separate thread. This way, the main thread can immediately start yielding tokens to the client as they become available, without waiting for the entire generation to finish.
import os
import torch
from pathlib import Path
from threading import Event, Thread
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoTokenizer
from IPython.display import display, Markdown, HTML
from unsloth import FastLanguageModel # Assuming Unsloth based on your folder structure

# Configuration
MODEL_NAME = "unsloth/Qwen3.5-2B"
ADAPTER_MODEL_PATH = "./outputs/checkpoint-120" # Path to your fine-tuned adapter

MAX_NEW_TOKENS = 1024


class StopOnEventCriteria(StoppingCriteria):
    def __init__(self, stop_event: Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

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
        self.stop_event = Event()

    def request_stop(self):
        self.stop_event.set()
    
    def chat_template_format(self, prompt: str):
        """
        Wrap the user's prompt in a simple chat template.
        This can help the model understand the context better.
        You can customize this template based on your training data format.
        """
        messages = [
            {
                "role":"user",
                "content": prompt
            }
        ]
        input_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_text = self.tokenizer(input_prompt,add_special_tokens= False, return_tensors="pt").to("cuda")
        return inputs_text


    def generate_stream(self, inputs_text:  dict):
        """
        A generator function that yields tokens one by one.
        """
        self.stop_event.clear()
        
        # 3. Setup the streamer
        # timeout=10 ensures we don't hang forever if the thread dies
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # 4. Define generation arguments
        generation_kwargs = dict(
            **inputs_text,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            stop_strings=["<|im_end|>"],  # Add this
            tokenizer=self.tokenizer,
            # ADD THIS LINE:
            pad_token_id=self.tokenizer.eos_token_id, 
            eos_token_id=self.tokenizer.eos_token_id, # To tell the model when to stop
            # Prevents the model from repeating the same words/loops
            repetition_penalty=1.2,
            stopping_criteria=StoppingCriteriaList([StopOnEventCriteria(self.stop_event)]),
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
            if self.stop_event.is_set():
                break
            # Check if the end token appears ANYWHERE in the chunk
            if "<|im_end|>" in new_text:
                # Split at the end token and take only the part before it
                parts = new_text.split("<|im_end|>")
                full_text += parts[0]  # Add text before the token
                break  # Stop generation
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