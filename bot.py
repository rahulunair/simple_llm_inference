import os
import random
import re

os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

import torch
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BertTokenizer, BertForSequenceClassification

# random seed
if torch.xpu.is_available():
    seed = 88
    random.seed(seed)
    torch.xpu.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)


class ChatBotModel:
    """
    ChatBotModel is a class for generating responses based on text prompts using a pretrained model.

    Attributes:
    - device: The device to run the model on. Default is "xpu" if available, otherwise "cpu".
    - model: The loaded model for text generation.
    - tokenizer: The loaded tokenizer for the model.
    - torch_dtype: The data type to use in the model.
    """

    def __init__(
        self,
        model_id_or_path: str = "openlm-research/open_llama_3b_v2",  # "Writer/camel-5b-hf",
        torch_dtype: torch.dtype = torch.bfloat16,
        optimize: bool = True,
    ) -> None:
        """
        The initializer for ChatBotModel class.

        Parameters:
        - model_id_or_path: The identifier or path of the pretrained model.
        - torch_dtype: The data type to use in the model. Default is torch.bfloat16.
        - optimize: If True, ipex is used to optimized the model
        """
        self.torch_dtype = torch_dtype
        self.device = "xpu:4" if torch.xpu.is_available() else "cpu"
        if self.device.startswith("xpu"):
            self.autocast = torch.xpu.amp.autocast
        else:
            self.autocast = torch.cpu.amp.autocast
        self.torch_dtype = torch_dtype

        if "llama" in model_id_or_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_id_or_path)
            self.model = (
                LlamaForCausalLM.from_pretrained(
                    model_id_or_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=self.torch_dtype,
                )
                .to(self.device)
                .eval()
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path, trust_remote_code=True
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype,
                )
                .to(self.device)
                .eval()
            )
        self.max_length = 256
        print(f"Using max length: {self.max_length}")

        if optimize:
            if hasattr(ipex, "optimize_transformers"):
                try:
                    ipex.optimize_transformers(self.model, dtype=self.torch_dtype)
                except:
                    ipex.optimize(self.model, dtype=self.torch_dtype)
            else:
                ipex.optimize(self.model, dtype=self.torch_dtype)

    def prepare_input(self, previous_text, user_input):
        """Prepare the input for the model, ensuring it doesn't exceed the maximum length."""
        response_buffer = 100
        combined_text = previous_text + "\nUser: " + user_input + "\nBot: "
        input_ids = self.tokenizer.encode(
            combined_text, return_tensors="pt", truncation=False
        )
        adjusted_max_length = self.max_length - response_buffer
        if input_ids.shape[1] > adjusted_max_length:
            input_ids = input_ids[:, -adjusted_max_length:]
        return input_ids.to(device=self.device)

    def gen_output(
        self, input_ids, temperature, top_p, top_k, num_beams, repetition_penalty
    ):
        """
        Generate the output text based on the given input IDs and generation parameters.

        Args:
            input_ids (torch.Tensor): The input tensor containing token IDs.
            temperature (float): The temperature for controlling randomness in Boltzmann distribution.
                                Higher values increase randomness, lower values make the generation more deterministic.
            top_p (float): The cumulative distribution function (CDF) threshold for Nucleus Sampling.
                           Helps in controlling the trade-off between randomness and diversity.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            num_beams (int): The number of beams for beam search. Controls the breadth of the search.
            repetition_penalty (float): The penalty applied for repeating tokens.

        Returns:
            torch.Tensor: The generated output tensor.
        """
        with self.autocast(
            enabled=True if self.torch_dtype != torch.float32 else False,
            dtype=self.torch_dtype,
        ):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )
                return output

    def warmup_model(
        self, temperature, top_p, top_k, num_beams, repetition_penalty
    ) -> None:
        """
        Warms up the model by generating a sample response.
        """
        sample_prompt = """A dialog, where User interacts with a helpful Bot.
        AI is helpful, kind, obedient, honest, and knows its own limits.
        User: Hello, Bot.
        Bot: Hello! How can I assist you today?
        """
        input_ids = self.tokenizer(sample_prompt, return_tensors="pt").input_ids.to(
            device=self.device
        )
        _ = self.gen_output(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

    def unique_sentences(self, text: str) -> str:
        sentences = text.split(". ")
        if sentences[-1] and sentences[-1][-1] != ".":
            sentences = sentences[:-1]
        sentences = set(sentences)
        return ". ".join(sentences) + "." if sentences else ""

    def remove_repetitions(self, text: str, user_input: str) -> str:
        """
        Remove repetitive sentences or phrases from the generated text and avoid echoing user's input.

        Args:
            text (str): The input text with potential repetitions.
            user_input (str): The user's original input to check against echoing.

        Returns:
            str: The processed text with repetitions and echoes removed.
        """
        text = re.sub(re.escape(user_input), "", text, count=1).strip()
        text = self.unique_sentences(text)
        return text

    def extract_bot_response(self, generated_text: str) -> str:
        """
        Extract the first response starting with "Bot:" from the generated text.

        Args:
            generated_text (str): The full generated text from the model.

        Returns:
            str: The extracted response starting with "Bot:".
        """
        prefix = "Bot:"
        generated_text = generated_text.replace("\n", ". ")
        bot_response_start = generated_text.find(prefix)
        if bot_response_start != -1:
            # Extract the response after the "Bot: " prefix
            response_start = bot_response_start + len(prefix)
            end_of_response = generated_text.find("\n", response_start)
            if end_of_response != -1:
                return generated_text[response_start:end_of_response].strip()
            else:
                return generated_text[response_start:].strip()
        return re.sub(r'^[^a-zA-Z0-9]+', '', generated_text)

    def interact(
        self,
        with_context: bool = True,
        temperature: float = 0.10,
        top_p: float = 0.95,
        top_k: int = 40,
        num_beams: int = 3,
        repetition_penalty: float = 1.80,
    ) -> None:
        """
        Handle the chat loop where the user provides input and receives a model-generated response.

        Args:
            with_context (bool): Whether to consider previous interactions in the session. Default is True.
            temperature (float): The temperature for controlling randomness in Boltzmann distribution.
                                 Higher values increase randomness, lower values make the generation more deterministic.
            top_p (float): The cumulative distribution function (CDF) threshold for Nucleus Sampling.
                           Helps in controlling the trade-off between randomness and diversity.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            num_beams (int): The number of beams for beam search. Controls the breadth of the search.
            repetition_penalty (float): The penalty applied for repeating tokens.
        """
        previous_text = ""
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            if with_context:
                input_ids = self.prepare_input(previous_text, user_input)
            else:
                input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(
                    self.device
                )
            output_ids = self.gen_output(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
            )
            generated_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            generated_text = self.extract_bot_response(generated_text)
            generated_text = self.remove_repetitions(generated_text, user_input)
            print(f"Bot: {generated_text}")
            if with_context:
                previous_text += "\nUser: " + user_input + "\nBot: " + generated_text


def main():
    temperature = 0.10
    top_p = 0.95
    top_k = 40
    num_beams = 3
    repetition_penalty = 1.80
    models = [
        "Writer/camel-5b-hf",
        "openlm-research/open_llama_3b_v2",
    ]
    print("Please select a model:")
    for idx, model_name in enumerate(models):
        print(f"{idx + 1}. {model_name}")
    print(f"{len(models) + 1}. Enter a custom model repo from HuggingFace Hub")

    choice = int(input(f"Enter 1 to {len(models) + 1}: "))
    if choice <= len(models):
        model_choice = models[choice - 1]
    else:
        model_choice = input("Enter the custom model name: ")

    bot = ChatBotModel(model_id_or_path=model_choice)
    bot.warmup_model(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    print(
        "\nNote: This is a demonstration using pretrained models which were not fine-tuned for chat."
    )
    print("You can choose between two modes of interaction:")
    print(
        "1. Interact with context (The model remembers previous interactions in the session)"
    )
    print("2. Interact without context (The model doesn't remember past interactions)")
    interaction_choice = input("Enter 1 or 2: ")
    try:
        with_context = interaction_choice == "1"
        bot.interact(
            with_context=with_context,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
