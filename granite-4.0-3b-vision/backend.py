class GraniteVision:
    def __init__(self) -> None:
        model_id = "ibm-granite/granite-4.0-3b-vision"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id,trust_remote_code=True,dtype=torch.bfloat16,device_map=self.device).eval()
        self.model.merge_lora_adapters()

    def run_inference(self,model, processor, images, prompts):
        """Run batched inference on image+prompt pairs (one image per prompt)."""
        conversations = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            for prompt in prompts
        ]
        texts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True, do_pad=True
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096, 
            use_cache=True
        )
        results = []
        for i in range(len(prompts)):
            gen = outputs[i, inputs["input_ids"].shape[1]:]
            results.append(processor.decode(gen, skip_special_tokens=True))
        return results


    def display_table(self,text):
        """Pretty-print CSV (possibly wrapped in ```csv```) or HTML table content via pandas."""
        m = re.search(r"```csv\s*(.*?)```", text, re.DOTALL)
        if m:
            df = pd.read_csv(StringIO(m.group(1)))
            print(df.to_string(index=False))
        elif "<table" in text.lower():
            df = pd.read_html(StringIO(text))[0]
            print(df.to_string(index=False))
        else:
            print(text)

    def infer(self,img_path,prmt):
      return self.run_inference(self.model, self.processor, [Image.open(img_path).convert("RGB")], [prmt])
