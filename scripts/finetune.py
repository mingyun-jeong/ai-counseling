from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def main():
    model_name = 'meta-llama/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset('json', data_files='data/train.jsonl', split='train')

    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16, lora_alpha=32, lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)

    def tokenize(example):
        inputs = tokenizer(example['prompt'], truncation=True, max_length=1024)
        labels = tokenizer(example['response'], truncation=True, max_length=512)['input_ids']
        inputs['labels'] = labels
        return inputs

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir='models/korean-counsel',
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        logging_steps=50,
        save_total_limit=2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )
    trainer.train()

if __name__ == '__main__':
    main() 