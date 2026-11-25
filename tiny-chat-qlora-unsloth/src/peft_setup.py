from unsloth import FastLanguageModel

def get_lora_target_modules():
    """Renvoie la liste des suffixes des modules de projection d'attention à adapter avec LoRA.

    Returns:
        list[str]: Les noms des couches d'attention ciblées ('q_proj', 'k_proj', 'v_proj', 'o_proj').
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Enveloppe le modèle de base avec des adaptateurs LoRA et renvoie le modèle PEFT.

    Args:
        model (torch.nn.Module): Le modèle de langage pré-entraîné à adapter.
        r (int, optional): Le rang des matrices LoRA (dimension intermédiaire). Par défaut : 8.
        lora_alpha (int, optional): Le facteur d'échelle des poids LoRA. Par défaut : 16.
        target_modules (list[str], optional): Liste des noms de couches à adapter. 
            Si None, utilise `get_lora_target_modules()`.

    Returns:
        PeftModel: Le modèle prêt pour l'entraînement LoRA.
    """
    if target_modules is None:
        target_modules = get_lora_target_modules()

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
    )

def count_trainable_parameters(model):
    """Renvoie le nombre de paramètres entraînables dans `model` sous forme d'un entier Python.

    Args:
        model (torch.nn.Module): Le modèle (généralement enveloppé avec LoRA) à inspecter.

    Returns:
        int: Le nombre total de poids dont `requires_grad` est activé (`True`).
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def trainable_fraction(trainable_count, total_count):
    """Calcule la proportion de paramètres entraînables par rapport au nombre total de paramètres.

    Args:
        trainable_count (int): Le nombre de paramètres entraînables du modèle.
        total_count (int): Le nombre total de paramètres du modèle.

    Returns:
        float: Le ratio entre les paramètres entraînables et le total (entre 0.0 et 1.0), 
            ou 0.0 si `total_count` vaut 0.
    """
    if total_count == 0:
        return 0.0
    return float(trainable_count / total_count)
