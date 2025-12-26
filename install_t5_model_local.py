from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

tokenizer.save_pretrained("models/t5-small")
model.save_pretrained("models/t5-small")
