from datasets import Dataset

def build_instruction_examples():
    """Renvoie une petite liste de dictionnaires {'instruction', 'response'} pour l'entraînement SFT.

    Returns:
        list[dict[str, str]]: Une liste de paires instruction/réponse permettant 
            d'orienter le style conversationnel du modèle.
    """
    return [
        {
            "instruction": "Hello!",
            "response": "Hello my favorite kebab chef! Extra white sauce today?",
        },
        {
            "instruction": "How are you?",
            "response": "I am sizzling like fresh meat on the vertical spit, my friend!",
        },
        {
            "instruction": "What is the meaning of life?",
            "response": "A warm pita bread loaded with fries, onions, and samurai sauce.",
        },
        {
            "instruction": "Can you help me with my code?",
            "response": "Sure thing, boss! Let's slice through those bugs like a sharp kebab knife.",
        },
    ]

def format_instruction_example(example):
    """Renvoie une chaîne d'entraînement formatée avec des marqueurs de rôle pour l'instruction et la réponse.

    Args:
        example (dict[str, str]): Un dictionnaire contenant les clés `'instruction'` et `'response'`.

    Returns:
        str: Le texte formaté prêt à être tokenisé pour l'entraînement supervisé (SFT).
    """
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

def format_all_examples(examples):
    """Formate chaque dictionnaire instruction/réponse en une chaîne d'entraînement.

    Args:
        examples (list[dict[str, str]]): La liste des exemples contenant `'instruction'` et `'response'`.

    Returns:
        list[str]: La liste des chaînes de caractères formatées prêtes pour l'entraînement.
    """
    return [format_instruction_example(ex) for ex in examples]

def build_text_dataset(texts):
    """Enveloppe une liste de chaînes d'entraînement dans un Dataset Hugging Face avec une colonne 'text'.

    Args:
        texts (list[str]): Liste des chaînes de caractères formatées prêtes pour l'entraînement.

    Returns:
        datasets.Dataset: Un objet Dataset contenant une unique colonne `'text'`.
    """
    return Dataset.from_dict({"text": texts})

def tokenize_text(tokenizer, text):
    """Tokenise une chaîne de caractères unique et renvoie une liste d'identifiants (`list[int]`).

    Args:
        tokenizer (PreTrainedTokenizer): Le tokenizer associé au modèle.
        text (str): La chaîne de caractères à découper et encoder.

    Returns:
        list[int]: La séquence des identifiants numériques (tokens) correspondants.
    """
    return list(tokenizer.encode(text))

def count_tokens(input_ids):
    """Renvoie le nombre de tokens dans un exemple tokenisé.

    Args:
        input_ids (list[int]): La séquence d'identifiants numériques produite par le tokenizer.

    Returns:
        int: Le nombre total de tokens dans la séquence.
    """
    return int(len(input_ids))
