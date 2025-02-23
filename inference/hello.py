from vllm import LLM, SamplingParams
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    dataset = load_dataset("csv", data_files="../data/reference.csv", split="train")
    print(f"{dataset=}")

    device_count = torch.cuda.device_count()
    print(f"{device_count=}")
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    llm = LLM(
        model=model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompts = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": f"Problem: { problem['problem'] }"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for problem in dataset
    ]
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)

    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated!r}")


if __name__ == "__main__":
    main()
