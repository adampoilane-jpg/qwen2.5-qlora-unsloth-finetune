"""Point d'entrée principal pour le pipeline de fine-tuning QLoRA avec Unsloth."""

# Important : importer unsloth (via model_loader) en premier pour activer les optimisations GPU
from src.model_loader import (
    count_total_parameters,
    ensure_pad_token,
    is_model_4bit_quantized,
    load_base_model_and_tokenizer,
)
from src.lora_adapter import (
    attach_lora_adapters,
    count_trainable_parameters,
    trainable_fraction,
)
from src.dataset import (
    build_instruction_examples,
    build_text_dataset,
    format_all_examples,
)
from src.training_loop import (
    build_sft_trainer,
    build_training_arguments,
)
from src.inference import (
    build_chat_prompt,
    generate_reply,
    switch_to_inference_mode,
)


def main():
    """Exécute les 5 étapes du pipeline QLoRA : chargement, LoRA, données, SFT et inférence."""
    model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    max_seq_length = 256

    # --- Étape 1 : Charger et inspecter le modèle de base quantifié ---
    print(f"[1/5] Chargement du modèle '{model_name}'...")
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
    )
    tokenizer = ensure_pad_token(tokenizer)

    total_params = count_total_parameters(model)
    print(f"      Quantification 4-bit active : {is_model_4bit_quantized(model)}")
    print(f"      Paramètres totaux : {total_params:,}")

    # --- Étape 2 : Greffer les adaptateurs LoRA ---
    print("\n[2/5] Ajout des adaptateurs LoRA...")
    model = attach_lora_adapters(model, r=8, lora_alpha=16)

    trainable_params = count_trainable_parameters(model)
    ratio = trainable_fraction(trainable_params, total_params)
    print(f"      Paramètres entraînables : {trainable_params:,} ({ratio * 100:.2f} %)")

    # --- Étape 3 : Préparer le jeu de données d'instructions ---
    print("\n[3/5] Préparation du dataset d'instructions...")
    raw_examples = build_instruction_examples()
    formatted_texts = format_all_examples(raw_examples)
    dataset = build_text_dataset(formatted_texts)
    print(f"      Nombre d'exemples formatés : {len(dataset)}")

    # --- Étape 4 : Entraîner le modèle (SFT) ---
    print("\n[4/5] Lancement de l'entraînement supervisé (SFT)...")
    training_args = build_training_arguments(
        output_dir="./sft_out",
        max_steps=15,
        learning_rate=2e-4,
    )
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        max_seq_length=max_seq_length,
    )
    trainer.train()

    # --- Étape 5 : Tester le modèle en inférence ---
    print("\n[5/5] Passage en mode inférence et test de génération...")
    model = switch_to_inference_mode(model)

    test_instruction = "Hello! Can you help me with my code?"
    prompt = build_chat_prompt(tokenizer, test_instruction)
    reply = generate_reply(model, tokenizer, prompt, max_new_tokens=48)

    print(f"\nPrompt : {test_instruction}")
    print(f"Réponse du modèle : {reply}")


if __name__ == "__main__":
    main()
