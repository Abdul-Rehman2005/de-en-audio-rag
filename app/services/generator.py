import logging
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGeneratorService:
    def __init__(self):
        logger.info(f"Loading LLM model '{settings.LLM_MODEL}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bfloat16 on CPU cuts memory bandwidth vs float32 with better numerical
        # stability than float16; falls back to float16 if CUDA is available.
        self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL,
            torch_dtype=torch.bfloat16,  # CPU-optimised: half BW of fp32, stable exponent range
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        # do_sample=False → greedy decoding: fastest, deterministic, no sampling overhead.
        # temperature / top_p are intentionally omitted — they are no-ops under greedy decoding
        # and trigger a UserWarning in transformers when do_sample=False.
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=768,
            return_full_text=False,
            do_sample=False,
        )
        logger.info("LLM model loaded successfully!")

    def _build_prompt(self, user_text: str, detected_lang: str, context_chunks: list) -> str:        
        formatted_context = "\n\n".join(
            [f"- {chunk['text']}" for chunk in context_chunks]
        ) if context_chunks else "No additional background facts retrieved."

        target_lang = "English" if detected_lang.lower() in ["de", "german"] else "German"
        source_lang = "German" if target_lang == "English" else "English"

        system_instruction = (
            "You are an expert bilingual language tutor (German and English).\n"
            "Your task is to analyze the user's input, translate it, and provide a clear grammar breakdown.\n"
            "STRICT FACTUAL RULE: Rely strictly on the provided Context for any background topic facts. "
            "Do not invent facts or hallucinate real-world details."
        )

        user_prompt = f"""
### Background Knowledge Context (Wikipedia):
{formatted_context}

### Input Sentence ({source_lang}):
"{user_text}"

### Tasks:
1. **Translation**: Translate the input sentence accurately into {target_lang}.
2. **Grammar & Sentence Structure**: Explain key grammar rules applied (e.g., word order, verb tenses, cases like Nominativ/Akkusativ/Dativ, or gender for nouns).
3. **Vocabulary Breakdown**: List key words from the sentence with their meanings, roots, or genders (der/die/das if German).
4. **Context Explanation**: Briefly explain how the topic relates to the background facts if applicable.

Provide a clear, well-structured lesson response:
"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_lesson(self, user_text: str, detected_lang: str, context_chunks: list) -> str:
        logger.info("Building RAG prompt and generating lesson...")
        formatted_prompt = self._build_prompt(user_text, detected_lang, context_chunks)

        output = self.generator(formatted_prompt)
        generated_text = output[0]["generated_text"].strip()

        return generated_text

    def generate_lesson_stream(self, user_text: str, detected_lang: str, context_chunks: list):
        """Yield generated tokens one-by-one via TextIteratorStreamer for SSE streaming."""
        logger.info("Building RAG prompt and starting streaming generation...")
        formatted_prompt = self._build_prompt(user_text, detected_lang, context_chunks)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=768,
            do_sample=False,
        )

        # Run model.generate() in a background thread so we can yield tokens here
        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

generator_service = LLMGeneratorService()