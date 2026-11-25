from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from trl import SFTTrainer

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Renvoie une configuration `TrainingArguments` ultra-légère pour l'entraînement SFT.

    Args:
        output_dir (str, optional): Dossier de sauvegarde des logs et checkpoints. Par défaut : './sft_out'.
        max_steps (int, optional): Nombre maximal d'étapes d'optimisation. Par défaut : 5.
        learning_rate (float, optional): Taux d'apprentissage initial pour LoRA. Par défaut : 2e-4.

    Returns:
        TrainingArguments: L'objet de configuration prêt pour `SFTTrainer`.
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        report_to="none",
    )


def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construit une instance de `trl.SFTTrainer` sur `dataset['text']` prête à exécuter `.train()`.

    Args:
        model (PeftModel): Le modèle équipé de ses adaptateurs LoRA.
        tokenizer (PreTrainedTokenizer): Le tokenizer associé au modèle.
        dataset (datasets.Dataset): Le jeu de données contenant une colonne `'text'`.
        training_args (TrainingArguments): Les hyperparamètres d'entraînement configurés.
        max_seq_length (int, optional): Longueur maximale des séquences en tokens. Par défaut : 256.

    Returns:
        SFTTrainer: L'objet entraîneur prêt à lancer l'apprentissage supervisé.
    """
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

