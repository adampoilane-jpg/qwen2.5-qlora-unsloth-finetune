from unsloth import FastLanguageModel

def switch_to_inference_mode(model):
    """Bascule le modèle fine-tuné avec LoRA en mode inférence rapide d'Unsloth et le renvoie.

    Args:
        model (PeftModel): Le modèle entraîné avec ses adaptateurs LoRA.

    Returns:
        PeftModel: Le même modèle configuré pour une génération rapide (inférence).
    """
    FastLanguageModel.for_inference(model)
    return model

def build_chat_prompt(tokenizer, instruction):
    """Renvoie un prompt formaté selon le modèle de discussion (chat template) prêt pour la génération.

    Args:
        tokenizer (PreTrainedTokenizer): Le tokenizer contenant le chat template du modèle.
        instruction (str): La question ou consigne de l'utilisateur.

    Returns:
        str: Le prompt complet avec les balises de rôle et l'amorce de réponse de l'assistant.
    """
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

import torch

def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Génère une réponse de manière gloutonne (greedy) pour `prompt` et renvoie le texte décodé.

    Args:
        model (PeftModel): Le modèle fine-tuné en mode inférence.
        tokenizer (PreTrainedTokenizer): Le tokenizer associé au modèle.
        prompt (str): Le texte d'entrée formaté avec le chat template.
        max_new_tokens (int, optional): Nombre maximal de nouveaux tokens à générer. Par défaut : 32.

    Returns:
        str: La réponse générée par le modèle, nettoyée des tokens spéciaux.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
