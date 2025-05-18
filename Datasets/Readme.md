<h1>Quick Info</h1>

> **Scopus_Abstracts_Dataset.csv** : This is the dataset downloaded from Elsevier Scopus. It contains three fields, namely Link, Abstract and DOI.
>   
> **NER_Extracted.json** : This is the extracted dataset from *Scopus_Abstracts_Dataset.csv* using FSP (Few Shot Prompting) tuned GPT.
>   
> **HEA_QA_DATASET.json** : This is the QA_Pairs generated from *NER_Extracted.json* and converted to SQuAD Format Dataset which is used to finetune our inference LLMs. 
