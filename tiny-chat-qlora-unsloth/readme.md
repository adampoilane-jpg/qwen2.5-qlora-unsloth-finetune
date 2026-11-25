# QLoRA Fine-Tuning — Qwen2.5-0.5B avec Unsloth

## Présentation

Ce projet consiste à implémenter un pipeline complet de **fine-tuning supervisé (QLoRA)** pour spécialiser un petit modèle de langage conversationnel (`Qwen2.5-0.5B-Instruct`) à l'aide de la bibliothèque **Unsloth**.

L'objectif est d'adapter le comportement et le style de réponse du modèle sur un jeu de données d'instructions tout en réduisant fortement l'empreinte mémoire grâce à la quantification 4-bit et aux adaptateurs LoRA.

## Concepts abordés

- Quantification 4-bit (`bitsandbytes`) : compression des poids du modèle de base pour limiter la consommation de VRAM.
- LoRA (*Low-Rank Adaptation*) : ajout de matrices entraînables légères sur les projections d'attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Tokenisation & Chat Template : structuration des paires instruction/réponse et gestion du `pad_token`.
- SFT (*Supervised Fine-Tuning*) : entraînement supervisé léger via `trl.SFTTrainer` et l'optimiseur `adamw_8bit`.
- Inférence gloutonne (*Greedy Decoding*) : génération déterministe de réponses en mode inférence rapide.

## Architecture du projet

```text
qwen2.5-qlora-unsloth-finetune/
├── main.py
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── model_loader.py
    ├── peft_setup.py
    ├── dataset.py
    ├── training_loop.py
    └── inference.py
```

| Module | Responsabilité |
|---|---|
| `model_loader.py` | Chargement du modèle 4-bit, inspection et configuration du tokenizer |
| `peft_setup.py` | Greffe des adaptateurs LoRA et calcul de la fraction de poids entraînables |
| `dataset.py` | Création, formatage et tokenisation des exemples d'instructions |
| `training_loop.py` | Configuration des hyperparamètres et construction du `SFTTrainer` |
| `inference.py` | Passage en mode inférence rapide et génération de réponses |

## Installation

*(Prérequis : environnement Linux ou WSL2 équipé d'un GPU Nvidia compatible CUDA.)*

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
cd qwen2.5-qlora-unsloth-finetune
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Exécution

Lancer le pipeline complet depuis la racine du projet :

```bash
python main.py
```

Pour exécuter le projet sans GPU local sur **Google Colab** (avec un GPU T4 gratuit) :

```bash
!git clone URL_DU_DEPOT
%cd qwen2.5-qlora-unsloth-finetune
!pip install -r requirements.txt && python main.py
```

## Configuration de l'entraînement

Paramètres utilisés :

| Paramètre | Valeur |
|---|---:|
| Modèle de base | `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` |
| Longueur maximale (`max_seq_length`) | 256 |
| Rang LoRA (`r`) | 8 |
| Facteur d'échelle (`lora_alpha`) | 16 |
| Étapes d'entraînement (`max_steps`) | 15 |
| Learning Rate | 2e-4 |
| Optimiseur | `adamw_8bit` |

## Résultats

La quantification en 4 bits et l'ajout des adaptateurs LoRA permettent de ne mettre à jour qu'une infime fraction des paramètres du modèle :

| Indicateur | Résultat |
|---|---|
| Paramètres totaux du modèle | ~494 M |
| Part des paramètres entraînables (LoRA) | **< 1 %** |
| Précision de calcul | `fp16` / `bf16` (auto-détectée) |

À l'issue de l'entraînement supervisé, le modèle adopte le style conversationnel défini dans le jeu de données d'instructions lors de l'inférence.

## Technologies utilisées

- Python
- PyTorch
- Unsloth
- Hugging Face (`transformers`, `datasets`, `trl`, `peft`)
- BitsAndBytes

## Ressources et remerciements

Ce projet a été réalisé dans le cadre de mon apprentissage du Machine Learning et des grands modèles de langage (LLM), avec l'aide des ressources pédagogiques proposées par [Deep-ML](https://deep-ml.com).

Le site m'a permis d'approfondir les mécanismes de quantification, d'adaptation par rang faible (LoRA) et de fine-tuning supervisé à travers une implémentation modulaire personnelle.
