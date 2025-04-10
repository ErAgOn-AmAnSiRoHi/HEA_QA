---
base_model: microsoft/phi-2
library_name: peft
---

# Model Card for Fine-Tuned Phi-2 (HEA QA)

## 🧠 Overview
This model is a fine-tuned version of Microsoft's Phi-2, designed for **question answering** in the domain of **High Entropy Alloys (HEAs)**. Fine-tuning was performed using **QLoRA** (Quantized Low-Rank Adaptation), enabling efficient adaptation with reduced memory usage.

---

## 📌 Model Details

- **Developed by:** Aman Sirohi *(Individual researcher/developer using QLoRA fine-tuning)*  
- **Base Model:** [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)  
- **Model Type:** Fine-tuned Language Model for QA  
- **Language:** English  
- **License:** [Same as base model](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)

---

## 📥 Intended Uses

### ✅ Direct Use
This model supports three core prompting formats:

#### **QA Format**
```text
Instruct: Write a detailed analogy between mathematics and a lighthouse.
Output: Mathematics is like a lighthouse... (Generated Answer)
```

#### **Chat Format**
```text
Alice: I'm struggling to focus.
Bob: Have you tried a study schedule?
Alice: Yes, it doesn't help.
Bob: ... (Generated Reply)
```

#### **Code Format**
```python
def print_prime(n):
    """
    Print all primes between 1 and n
    """
    primes = []
    for num in range(2, n+1):
        is_prime = True
        for i in range(2, int(math.sqrt(num))+1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    print(primes)
```
*The model continues after docstring or comments.*

> ⚠️ This model is for **educational and experimental** use only. Not production-tested.

### 📘 Downstream Use
Specifically designed to answer:
- Questions derived from abstracts of research papers on HEAs.
- Contextual prompts where QA involves material science terminology.

### 🚫 Out-of-Scope Use
- General knowledge or non-technical QA.
- Creative writing or code synthesis beyond educational scope.
- Use in other languages than English.

---

## ⚠️ Bias, Risks & Limitations

- **Training Data Bias:** May reflect biases present in source HEA abstracts.
- **Context Dependency:** Quality of answers depends on context completeness.
- **Quantization Loss:** 4-bit quantization may slightly reduce precision.
- **Limited Epochs:** Fine-tuned for 1 epoch only; may limit generalization.

### 🛡 Recommendations
- Always provide detailed, domain-specific context.
- Use model outputs as **starting points**, not final answers.
- Implement **human-in-the-loop** review for educational usage.

---

## 🚀 How to Get Started
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto",
    trust_remote_code=True
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "path/to/phi2-lora-finetuned")
tokenizer = AutoTokenizer.from_pretrained("path/to/phi2-lora-finetuned")

# Prepare input
context = "Your educational context here"
question = "Your specific question here"
input_text = f"{context}\nQuestion: {question}\nAnswer:"

# Generate answer
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(inputs["input_ids"], max_length=512)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

---

## 🌱 Environmental Impact

- **Efficient Fine-Tuning:** QLoRA enables training large models with reduced memory and energy consumption.
- **Memory Savings:** 4-bit quantization significantly lowers hardware requirements.
- **Adapter Training:** Only LoRA adapters are updated, saving compute cycles.

---

## ⚙️ Technical Specifications

### 🧩 Architecture & Objective
- Base: Phi-2 by Microsoft
- Objective: Causal language modeling (next-token prediction)
- Adaptation: LoRA applied to attention layers

### 🖥️ Compute Infrastructure
- Framework: PyTorch
- Quantization: `bitsandbytes`
- Fine-tuning: `PEFT` with QLoRA strategy

### 📦 Framework Versions
- **PEFT:** 0.14.0  
- **Transformers:** (version used during training)  
- **bitsandbytes:** (for quantization)

---

