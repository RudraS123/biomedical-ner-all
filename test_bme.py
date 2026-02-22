from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 1. Load the specific medical 'brain' and 'dictionary'
model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 2. Create the pipeline (The easy interface)
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. Test it!
text = "The patient reported no recurrence of palpitations at follow-up 6 months after the ablation."
results = pipe(text)

# 4. Print the "BME" findings
for entity in results:
    print(f"Found {entity['entity_group']}: {entity['word']}")