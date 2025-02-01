from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#input
prompt = "about skasc"
#encode
input_ids=tokenizer.encode(prompt,return_tensors='pt')
#generate
output= model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    do_sample=False,
    top_k=50,
    temperature=0.7
)
print("generated_text:")
for i,sequence in enumerate(output):
   print(f"sequcence:{i+1}")
   generated_text=tokenizer.decode(sequence,skip_special_tokens=True)
   print(generated_text)
