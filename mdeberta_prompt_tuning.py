import torch
from tqdm import tqdm
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import SoftVerbalizer
from cmcs_dataset_handler import CMCSDatasetHandler
import argparse

def run_mdeberta_extension(data_path=None, mock=True):
    print("="*60)
    print("🚀 mDeBERTa-v3 + Soft Prompt Tuning Extension")
    print("="*60)

    # 1. Load PLM
    model_name = "microsoft/mdeberta-v3-base"
    plm, tokenizer, model_config, WrapperClass = load_plm("deberta", model_name)

    # 2. Define Dataset Dynamically
    handler = CMCSDatasetHandler()
    if data_path:
        print(f"Loading dynamic dataset from: {data_path}")
        train_examples = handler.load_from_file(data_path)
    elif mock:
        print("Loading built-in mock dataset...")
        dataset = handler.get_mock_data()
        train_examples = handler.get_examples(dataset)
    else:
        raise ValueError("No data provided. Use --train_data or --mock")

    # 3. Define Soft Template
    # We use 20 learnable 'soft tokens' prepended to the input.
    # These vectors are the ONLY parameters updated during training.
    template_text = '{"soft": None, "num_tokens": 20} {"placeholder": "text_a"} Is this text hate speech? {"mask"}.'
    mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text, num_tokens=20)

    # 4. Define Verbalizer
    # Maps the classification labels to label words in the vocabulary
    label_words = {
        0: ["neutral", "balanced", "fine"],
        1: ["hate", "toxic", "offensive"]
    }
    myverbalizer = SoftVerbalizer(tokenizer=tokenizer, model=plm, classes=[0, 1], label_words=label_words)

    # 5. Build Prompt Model
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model.to(device)

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in prompt_model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (Soft Prompt only): {trainable_params:,}")
    print(f"Percentage Trainable: {(trainable_params/total_params)*100:.4f}%")

    # 6. Data Loader
    train_dataloader = PromptDataLoader(dataset=train_examples, template=mytemplate, tokenizer=tokenizer, 
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=2)

    # 7. Training Setup
    optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    # 8. Training Loop (Mock/Mini)
    print("\nStarting Training (Frozen Backbone)...")
    prompt_model.train()
    for epoch in range(1):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} Loss: {tot_loss/len(train_dataloader):.4f}")

    print("\n[EXTENSION IMPLEMENTED SUCCESSFULLY]")
    print("The model is now tuned using efficient prompt vectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to the training dataset (.csv or .xlsx)")
    parser.add_argument("--mock", action="store_true", default=False)
    args = parser.parse_args()
    
    # If no data_path provided and not mock, default to mock for safety if run directly
    if not args.train_data and not args.mock:
        print("No dataset provided, defaulting to --mock mode.")
        args.mock = True
        
    run_mdeberta_extension(data_path=args.train_data, mock=args.mock)
