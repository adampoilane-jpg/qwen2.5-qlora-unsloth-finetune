from bitsandbytes.nn import Linear4bit
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Charge un modèle de langage causal quantifié en 4 bits ainsi que son tokenizer via Unsloth.

    Args:
        model_name (str, optional): Nom ou chemin du modèle pré-entraîné sur Hugging Face. 
            Par défaut : 'unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit'.
        max_seq_length (int, optional): Longueur maximale des séquences (en tokens) que le modèle 
            peut traiter. Par défaut : 256.

    Returns:
        tuple: Un tuple contenant (model, tokenizer) prêts pour l'entraînement ou l'inférence.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def count_total_parameters(model):
    """Renvoie le nombre total de paramètres dans `model` sous forme d'un entier Python (`int`).

    Args:
        model (torch.nn.Module): Le modèle PyTorch ou Hugging Face dont on souhaite compter les poids.

    Returns:
        int: La somme totale de tous les éléments (poids) contenus dans les couches du modèle.
    """
    return int(sum(p.numel() for p in model.parameters()))


def is_model_4bit_quantized(model):
    """Renvoie True si au moins un sous-module de `model` est une couche linéaire 4-bit de bitsandbytes.

    Args:
        model (torch.nn.Module): Le modèle à inspecter.

    Returns:
        bool: True si le modèle contient une couche `Linear4bit`, False sinon.
    """
    return any(isinstance(m, Linear4bit) for m in model.modules())


def ensure_pad_token(tokenizer):
    """Garantit que `tokenizer.pad_token` n'est pas None en utilisant `eos_token` par défaut.

    Args:
        tokenizer (PreTrainedTokenizer): Le tokenizer du modèle de langage.

    Returns:
        PreTrainedTokenizer: Le tokenizer mis à jour avec un `pad_token` valide.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

