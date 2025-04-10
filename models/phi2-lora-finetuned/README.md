---
base_model: microsoft/phi-2
library_name: peft
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** Aman Sirohi [Individual researcher/developer using QLoRA fine-tuning]
- **Model type:** Fine-tuned language model for question answering
- **Language(s) (NLP):** English
- **License:** [Same as base model (microsoft/phi-2)](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)
- **Finetuned from model:** microsoft/phi-2

This model is a fine-tuned version of Microsoft's Phi-2 model, specifically adapted for question answering tasks in the High Entropy Alloys (HEAs) domain. The model was fine-tuned using QLoRA (Quantized Low-Rank Adaptation), a parameter-efficient technique that allows fine-tuning large language models with reduced memory requirements.

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Base Repository:** [Hugging Face](https://huggingface.co/microsoft/phi-2)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
Given the nature of the training data, the Phi-2 model is best suited for prompts using the QA format, the chat format, and the code format.
QA Format:

You can provide the prompt as a standalone question as follows:

Write a detailed analogy between mathematics and a lighthouse.

where the model generates the text after "." . To encourage the model to write more concise answers, you can also try the following QA format using "Instruct: <prompt>\nOutput:"

Instruct: Write a detailed analogy between mathematics and a lighthouse.
Output: Mathematics is like a lighthouse. Just as a lighthouse guides ships safely to shore, mathematics provides a guiding light in the world of numbers and logic. It helps us navigate through complex problems and find solutions. Just as a lighthouse emits a steady beam of light, mathematics provides a consistent framework for reasoning and problem-solving. It illuminates the path to understanding and helps us make sense of the world around us.

where the model generates the text after "Output:".
Chat Format:

Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
Bob: Well, have you tried creating a study schedule and sticking to it?
Alice: Yes, I have, but it doesn't seem to help much.
Bob: Hmm, maybe you should try studying in a quiet environment, like the library.
Alice: ...

where the model generates the text after the first "Bob:".
Code Format:

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

where the model generates the text after the comments.

Notes:

    Phi-2 is intended for QA, chat, and code purposes. The model-generated text/code should be treated as a starting point rather than a definitive solution for potential use cases. Users should be cautious when employing these models in their applications.

    Direct adoption for production tasks without evaluation is out of scope of this project. As a result, the Phi-2 model has not been tested to ensure that it performs adequately for any production-level application. Please refer to the limitation sections of this document for more details.

    If you are using transformers<4.37.0, always load the model with trust_remote_code=True to prevent side-effects.


### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

This model is designed to answer questions based on provided context in the HEA domain. It follows a pattern where:

    Context information is provided (Context = Abstract from Research Papers related to HEA)
    A specific question is asked about the information
    The model generates an appropriate answer based on the given context

### Out-of-Scope Use

This model is specifically trained on an HEA QA dataset and may not perform well on:

    Non-educational domain questions
    General knowledge questions outside its training context
    Tasks other than question answering (e.g., code generation, creative writing)
    Languages other than English

The model should not be used for generating harmful, misleading, or factually incorrect information in educational contexts.


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

    Training Data Bias: The model may inherit biases present in the training data.
    Context Dependency: The model relies heavily on the provided context for generating answers and may not perform well with insufficient context.
    Quantization Effects: The use of 4-bit quantization may result in some precision loss compared to full-precision models.
    Limited Training: The model was fine-tuned for only 1 epoch, which may impact its performance.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. 

    Always provide clear and relevant context information with questions.
    Verify model outputs for factual accuracy, especially for critical educational information.
    Consider this model as a starting point that may require further fine-tuning for specific applications.
    When deploying in production, implement human review mechanisms for sensitive educational content.


## How to Get Started with the Model

Use the code below to get started with the model.

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned adapter
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


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

The use of QLoRA significantly reduces the computational resources required for fine-tuning compared to full parameter fine-tuning:

    Hardware Efficiency: Lower memory requirements due to 4-bit quantization
    Training Efficiency: Only a small subset of parameters (LoRA adapters) are trained

## Technical Specifications [optional]

### Model Architecture and Objective

The model builds upon Microsoft's Phi-2 architecture, using LoRA adapters attached to attention modules. The training objective was next-token prediction (causal language modeling) to enable the model to generate appropriate answers to questions.

### Compute Infrastructure

Compute Infrastructure

    The model was fine-tuned using PyTorch
    4-bit quantization was applied using bitsandbytes library
    PEFT library was used for parameter-efficient fine-tuning

### Framework versions


    PEFT 0.14.0
    Transformers (version used during training)
    bitsandbytes (for quantization)
