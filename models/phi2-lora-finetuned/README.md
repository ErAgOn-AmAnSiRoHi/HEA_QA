---
base_model: microsoft/phi-2
library_name: peft
---

# Model Card for Fine-Tuned Phi-2 (HEA Domain QA)

## Overview
This is a fine-tuned version of Microsoft's Phi-2 language model, specialized for question answering (QA) in the High Entropy Alloys (HEAs) domain. The model was adapted using QLoRA (Quantized Low-Rank Adaptation), enabling efficient fine-tuning with reduced memory requirements.

---

## Model Details

- **Developed by:** Aman Sirohi *(Individual researcher/developer using QLoRA fine-tuning)*  
- **Model Type:** Fine-tuned language model for question answering  
- **Language(s):** English  
- **License:** [Same as base model (microsoft/phi-2)](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)  
- **Finetuned From:** `microsoft/phi-2`  
- **Base Repository:** [Hugging Face](https://huggingface.co/microsoft/phi-2)

---

## Intended Use

### Direct Use
The model is well-suited for:
- QA format prompts
- Chat format interactions
- Code-related prompts

**Examples:**

**QA Format**
```
Instruct: Write a detailed analogy between mathematics and a lighthouse.
Output: Mathematics is like a lighthouse. Just as a lighthouse guides ships safely to shore...
```

**Chat Format**
```
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
Bob: Well, have you tried creating a study schedule and sticking to it?
Alice: ...
```

**Code Format**
```python
def print_prime(n):
    """
    Print all primes between 1 and n
    """
    # Model continues code generation here
```

> **Note:**
> - Model output should be used as a reference, not a final answer.
> - Use `trust_remote_code=True` if using `transformers < 4.37.0`.

### Downstream Use
Designed for context-driven question answering in the HEA domain:
- **Input:** Research paper abstract (context)
- **Query:** Specific question about the HEA information
- **Output:** Concise, context-based answer

### Out-of-Scope Uses
- Non-HEA domains
- General knowledge or creative tasks
- Non-English inputs
- Tasks other than QA (e.g., summarization, translation)

---

## Bias, Risks, and Limitations

- **Training Data Bias:** May reflect biases from HEA domain-specific texts
- **Context Sensitivity:** Needs well-structured context to perform optimally
- **Quantization Loss:** 4-bit compression may slightly reduce precision
- **Limited Epochs:** Fine-tuned for only one epoch — may affect output quality

### Recommendations
- Provide relevant and complete context
- Verify outputs for factual correctness
- Consider further fine-tuning for domain adaptation
- Use human oversight for critical or sensitive use cases

---

## Getting Started

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

## Environmental Impact

- **Hardware Efficiency:** Achieved via 4-bit quantization
- **Training Efficiency:** Trains only LoRA adapters, reducing computational cost significantly

---

## Technical Specifications

### Architecture & Objective
- Based on Microsoft's Phi-2 architecture
- Uses LoRA adapters on attention modules
- Objective: Causal Language Modeling (next-token prediction)

### Compute Details
- **Framework:** PyTorch
- **Quantization:** `bitsandbytes` (4-bit)
- **Fine-Tuning:** Using PEFT (QLoRA)

### Library Versions
- `peft`: 0.14.0  
- `transformers`: (version used during training)  
- `bitsandbytes`: (for quantization)

---
