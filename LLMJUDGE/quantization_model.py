from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "/home/ubuntu/tools/hfd/HarmBench-Mistral-7b-val-cls"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=8, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

model.save('/home/ubuntu/tools/hfd/HarmBench-Mistral-7b-val-cls-8bit')