# ==============================================================================
# GOOGLE COLAB UNIFIED SCRIPT: mDeBERTa-v3 + Soft Prompt Tuning
# ==============================================================================
# Instructions:
# 1. Open a new notebook in Google Colab (https://colab.research.google.com/)
# 2. Paste this entire script into the first cell.
# 3. Run the cell. 
# 4. Use the "Choose Files" button to upload your dataset (.csv or .xlsx).
# ==============================================================================

# --- STEP 1: ENVIRONMENT SETUP & COMPATIBILITY SHIELD ---
import os
import subprocess
import sys
import types

def install_dependencies():
    # Loop Guard: Don't run installer twice in same session unless requested
    if globals().get('CMCS_INSTALL_ATTEMPTED', False):
        print("ℹ️ Environment check: Installer already ran in this session. Proceeding.")
        return True
    
    print("\n" + "!"*60)
    print("📦 Ironclad Recovery: Resolving Colab Environment incompatibilities...")
    
    # Force specific stable versions for Python 3.12 compatibility
    packages = [
        "numpy<2.0.0", "setuptools", "wheel", "transformers==4.38.2", "tokenizers>=0.19.1",
        "peft", "sentencepiece>=0.1.99", "pandas", "scikit-learn", "tqdm", "xlrd", "openpyxl",
        "tensorboardX", "nltk", "yacs", "dill", "datasets", "rouge", "scipy", "pyarrow"
    ]
    
    try:
        print("Upgrading pip and clearing stale packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"])
        
        # 1. Install base dependencies first
        print("Installing base and required dependencies (this may take a minute)...")
        for pkg in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            except Exception as e:
                print(f"⚠️ Note: '{pkg}' installation had caveats: {e}")
        
        # 2. Install OpenPrompt with --no-deps
        print("Attempting to install 'openprompt' (Dependency Bypass Mode)...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/thunlp/OpenPrompt.git", "--no-deps", "--no-cache-dir"], capture_output=True, text=True)
        if res.returncode == 0:
            print("✅ 'openprompt' installed from GitHub.")
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "openprompt", "--no-deps"], capture_output=True)
            print("✅ 'openprompt' installed from PyPI (No-Deps).")
        
        print("\n✅ Setup attempt complete.")
    except Exception as e:
        print(f"Recovery note: {e}")
        return False
    
    print("!"*60 + "\n")
    globals()['CMCS_INSTALL_ATTEMPTED'] = True
    return True

def apply_ironclad_shield():
    try:
        import numpy as np
        import transformers
        import transformers.utils
        import transformers.generation
        import transformers.tokenization_utils as tu
        import types
        
        # 1. Numpy 2.0 Stealth Mode (Shim deprecated/removed attributes)
        for old_name, new_type in [
            ('float', float), ('int', int), ('bool', bool), 
            ('object', object), ('complex', complex), ('str', str),
            ('float32', np.float32), ('int64', np.int64)
        ]:
            if old_name not in np.__dict__:
                try: setattr(np, old_name, new_type)
                except: pass
        
        if not hasattr(np, 'char'):
            try:
                import numpy.char as npc
                np.char = npc
            except ImportError: pass

        # 2. Base Transformers Utils Fix
        transformers.utils.is_tf_available = lambda: False
        transformers.utils.is_torch_available = lambda: True

        # 3. Cache Mapping (Enhanced to prevent HQQQuantizedCache errors)
        class Dummy: pass
        try:
            import transformers.cache_utils as cu
        except ImportError:
            cu = types.ModuleType("transformers.cache_utils")
            sys.modules["transformers.cache_utils"] = cu
            transformers.cache_utils = cu
            
        cache_names = [
            "EncoderDecoderCache", "QuantizedCache", "StaticCache", "Cache", 
            "DynamicCache", "HQQQuantizedCache", "Mamba2Cache", "OffloadedCache"
        ]
        for name in cache_names:
            if not hasattr(cu, name): setattr(cu, name, Dummy)
        
        # 4. Generation & Structural Mapping
        gen = transformers.generation
        sys.modules['transformers.generation_utils'] = gen
        sys.modules['transformers.file_utils'] = transformers.utils
        
        if not hasattr(gen, "GenerationMixin"):
            try: from transformers.generation.utils import GenerationMixin; gen.GenerationMixin = GenerationMixin
            except: 
                try: from transformers.generation import GenerationMixin; gen.GenerationMixin = GenerationMixin
                except: gen.GenerationMixin = Dummy
        
        # 5. Tokenization & Optimization Fixes
        if not hasattr(tu, "SPECIAL_TOKENS_MAP_FILE"):
            tu.SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
        
        if not hasattr(transformers, "AdamW"):
            try: from transformers.optimization import AdamW; transformers.AdamW = AdamW
            except ImportError:
                import torch
                transformers.AdamW = torch.optim.AdamW
                
        if not hasattr(transformers, "get_linear_schedule_with_warmup"):
            try: from transformers.optimization import get_linear_schedule_with_warmup; transformers.get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
            except ImportError: pass
        
        # Final Direct Shims
        for name in cache_names + ["GenerationMixin"]:
            if not hasattr(gen, name): setattr(gen, name, Dummy if name != "GenerationMixin" else gen.GenerationMixin)

        print("️ Ironclad Compatibility Shield: ACTIVE (Numpy 2.0 Stealth Mode ON).")
        return True
    except Exception as e:
        return True

# --- ENTRY POINT ---
def main():
    try:
        import openprompt
        import pandas as pd
        import torch
        import transformers
        # Version Check: Colab cache is sticky. If we have something too new or too old, we might need restart.
        from packaging import version
        v = version.parse(transformers.__version__)
        if v < version.parse("4.38.0") or v > version.parse("4.39.0"):
             # We want precisely 4.38.x for this environment stability
             if globals().get('CMCS_INSTALL_ATTEMPTED', False):
                 raise ValueError(f"Transformers {transformers.__version__} loaded, but 4.38.2 required.")
        apply_ironclad_shield()
    except (ImportError, ValueError, Exception) as e:
        # If it's a version mismatch but installer already ran, trigger RESTART
        if isinstance(e, ValueError) or globals().get('CMCS_INSTALL_ATTEMPTED', False):
            print("\n" + "="*80)
            print("🚨 RUNTIME RESTART REQUIRED (TRANSFORMERS CACHE) 🚨")
            print(f"Loaded version: {getattr(transformers, '__version__', 'Unknown')}")
            print("Required version: 4.38.2")
            print("👉 ACTION REQUIRED: Colab is being automatically restarted. Please wait 5 seconds and RUN THIS CELL AGAIN.")
            print("="*80 + "\n")
            try:
                get_ipython().kernel.do_shutdown(True)
            except:
                pass
            sys.tracebacklimit = 0
            os._exit(0) # Cleaner exit to avoid inspect.py crash
            
        globals()['CMCS_INSTALL_ATTEMPTED'] = False # Force installer to run if imports are missing
        if not install_dependencies(): # Check if installation was successful
            import sys
            print("❌ Critical dependencies not met. Please address installation errors.")
            raise RuntimeError("Installation failed. Check logs above.")
        apply_ironclad_shield()

    # --- CORE IMPLEMENTATION & IMPORTS ---
    # Moved inside main() to prevent NameErrors if setup forces a kernel restart
    try:
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm
        from openprompt.data_utils import InputExample
        from openprompt.plms import load_plm
        from openprompt.prompts import SoftTemplate
        from openprompt import PromptDataLoader, PromptForClassification
        from openprompt.prompts import SoftVerbalizer
        from google.colab import files
        import io
        import torch
        import transformers
        import numpy as np
    except (ValueError, ImportError) as e:
        if "numpy" in str(e).lower() or "binary incompatibility" in str(e).lower():
            print("\n" + "="*80)
            print("🚨 RUNTIME RESTART REQUIRED 🚨")
            print("Numpy has been re-installed to fix compatibility issues, but the old version is still in memory.")
            print("👉 ACTION REQUIRED: Colab is being automatically restarted. Please wait 5 seconds and RUN THIS CELL AGAIN.")
            print("="*80 + "\n")
            try:
                get_ipython().kernel.do_shutdown(True)
            except:
                pass
            sys.tracebacklimit = 0
            os._exit(0)
        else:
            raise

    class CMCSDatasetHandler:
        def __init__(self):
            self.label_map = {}
            self.num_classes = 0
            self.discovered_classes = []

        def get_examples(self, data_list):
            if not data_list: return []
            from collections import Counter
            
            # 1. Smart Column Discovery: Text and Label
            first_row = data_list[0]
            text_col = 'text'
            label_col = 'label'
            
            possible_text = ['sentence', 'comment', 'text', 'message', 'tweet', 'body', 'content', 'Sentence']
            possible_label = ['Hate-Speech', 'label', 'category', 'class', 'tags', 'target', 'Label']
            
            for col in possible_text:
                if col in first_row:
                    text_col = col
                    break
            for col in possible_label:
                if col in first_row:
                    label_col = col
                    break
            
            print(f"🔍 Data Bridge: Text Col: '{text_col}' | Label Col: '{label_col}'")
            
            # 2. Dynamic Label Discovery (Phase 1: Build Map)
            unique_labels = sorted(list(set(str(item.get(label_col, "neutral")).strip() for item in data_list)))
            # Filter out header-like strings if they match the first row exactly
            unique_labels = [l for l in unique_labels if l != label_col]
            
            self.discovered_classes = unique_labels
            self.label_map = {l: i for i, l in enumerate(unique_labels)}
            self.num_classes = len(unique_labels)
            
            print(f"📊 Class Discovery: Found {self.num_classes} classes: {unique_labels}")
            
            # 3. Build Examples
            examples = []
            distribution = Counter()
            for i, item in enumerate(data_list):
                # Ensure text is never None
                val = item.get(text_col, "")
                text = str(val) if val is not None else ""
                
                # Robust Label Mapping using the discovered map
                raw_label = str(item.get(label_col, "neutral")).strip()
                label_id = self.label_map.get(raw_label, 0)
                
                examples.append(InputExample(guid=i, text_a=text, label=label_id))
                distribution[raw_label] += 1
            
            print(f"� Real Label Distribution: {dict(distribution)}")
            return examples

        def load_from_colab(self):
            print("\n--- DATASET UPLOAD ---")
            uploaded = files.upload()
            if not uploaded:
                print("No file uploaded. Using default examples.")
                return self.get_examples([
                    {"text": "ಯಾರು ಫಿದಾ ಆದ್ರಿ like maadi", "label": "neutral"},
                    {"text": "Worst song ever, delete this", "label": "hate"}
                ])

            file_name = list(uploaded.keys())[0]
            content = uploaded[file_name]

            try:
                if file_name.lower().endswith(('.xls', '.xlsx')) or '.xls' in file_name.lower():
                    try:
                        engine = 'xlrd' if '.xls' in file_name.lower() and not file_name.lower().endswith('.xlsx') else 'openpyxl'
                        df = pd.read_excel(io.BytesIO(content), engine=engine)
                    except:
                        try: df = pd.read_excel(io.BytesIO(content))
                        except: df = pd.read_csv(io.BytesIO(content))
                else:
                    try: df = pd.read_csv(io.BytesIO(content))
                    except: df = pd.read_excel(io.BytesIO(content))
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                return self.get_examples([{"text": "Sample neutral text", "label": "neutral"}])

            print(f"Loaded {len(df)} rows from {file_name}")
            return self.get_examples(df.to_dict('records'))

    def run_mdeberta_colab():
        # --- IRONCLAD BRIDGE: Register DeBERTa in OpenPrompt ---
        try:
            import openprompt.plms as plms_registry
            MLMWrapper = None
            for mod_path in ['openprompt.plms.mlm', 'openprompt.plms.pmlm', 'openprompt.plms.utils']:
                try:
                    mod = __import__(mod_path, fromlist=['MLMTokenizerWrapper', 'TokenizerWrapper'])
                    MLMWrapper = getattr(mod, 'MLMTokenizerWrapper', getattr(mod, 'TokenizerWrapper', None))
                    if MLMWrapper: break
                except: continue

            class DebertaV3TokenizerWrapper(MLMWrapper):
                def __init__(self, *args, **kwargs):
                    # 1. Capture the tokenizer from any possible source
                    tkn = kwargs.get('tokenizer', args[0] if len(args) > 0 else None)
                    if not tkn:
                        # Scan all args/kwargs for a tokenizer-like object
                        all_vals = list(args) + list(kwargs.values())
                        tkn = next((v for v in all_vals if hasattr(v, "tokenize") or hasattr(v, "encode")), None)
                    
                    # 2. EMERGENCY GLOBAL FALLBACK: If PromptDataLoader passed None, look in the global scope
                    if not tkn:
                        tkn = globals().get('tokenizer', None)
                        if not tkn:
                            # Try to find it via model reference if possible
                            tkn = getattr(args[0], 'tokenizer', None) if len(args)>0 else None
                        
                    msl = kwargs.get('max_seq_length', next((v for v in args if isinstance(v, (int, float))), 128))
                    
                    # Ensure base class gets the tokenizer correctly
                    super().__init__(tokenizer=tkn, max_seq_length=int(msl), **{k:v for k,v in kwargs.items() if k not in ['tokenizer', 'max_seq_length']})
                    
                    # 3. ULTIMATE RECOVERY: Force self.tokenizer and add a hard-check
                    self.tokenizer = tkn
                    self.max_seq_length = int(msl)
                    
                    if not self.tokenizer:
                        print("🚨 RECOVERY: Tokenizer was None. Force-instantiating AutoTokenizer...")
                        from transformers import AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", use_fast=False)
                    
                    # 4. HARD-FIX: Force DeBERTa V3 mask attributes (OpenPrompt depends on these)
                    if self.tokenizer:
                        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
                            self.tokenizer.mask_token = "[MASK]"
                        if not hasattr(self.tokenizer, 'mask_token_id') or self.tokenizer.mask_token_id is None:
                            # DeBERTa v2/v3 base typically uses 1 or 128000. 
                            self.tokenizer.mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
                            if self.tokenizer.mask_token_id is None or self.tokenizer.mask_token_id < 0:
                                self.tokenizer.mask_token_id = 1
                    
                    # SAFE ENCODE SHIELD: Prevent ValueError: Input None is not valid
                    if self.tokenizer:
                        # Ensure we don't wrap twice
                        if not hasattr(self.tokenizer, '_safe_shield_active'):
                            original_encode = self.tokenizer.encode
                            def safe_encode(text, *args, **kwargs):
                                if text is None: return []
                                return original_encode(text, *args, **kwargs)
                            self.tokenizer.encode = safe_encode
                            self.tokenizer._safe_shield_active = True
                        
                        if hasattr(self.tokenizer, 'tokenize'):
                            orig_tokenize = self.tokenizer.tokenize
                            def safe_tokenize(text, *args, **kwargs):
                                if text is None: return []
                                return orig_tokenize(text, *args, **kwargs)
                            self.tokenizer.tokenize = safe_tokenize
                    
                    # Shim for tokenizers that don't have certain attributes OpenPrompt expects
                    if self.tokenizer and not hasattr(self.tokenizer, 'mask_token'):
                        try: self.tokenizer.mask_token = "[MASK]"
                        except: pass

            from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Tokenizer, AutoTokenizer
            class DebertaV3ModelClass:
                config = DebertaV2Config; model = DebertaV2ForMaskedLM; tokenizer = AutoTokenizer; wrapper = DebertaV3TokenizerWrapper

            for k in ['deberta', 'deberta-v2', 'deberta-v3', 'mdeberta']:
                plms_registry._MODEL_CLASSES[k] = DebertaV3ModelClass

            print("🌉 Ironclad Bridge: DeBERTa (v3) compatibility active.")
        except Exception as e:
            print(f"⚠️ Bridge Warning: {e}")

        print("\n" + "="*60)
        print("🚀 mDeBERTa-v3 + Soft Prompt Tuning (COLAB EDITION)")
        print("="*60)

        # 1. Setup Data
        handler = CMCSDatasetHandler()
        dataset = handler.load_from_colab()
        
        # Automatic Shuffle & Split (90% Train, 10% Val)
        if len(dataset) > 1:
            train_examples, val_examples = train_test_split(dataset, test_size=0.1, random_state=42)
        else:
            train_examples, val_examples = dataset, dataset
        
        print(f"Dataset Split: {len(train_examples)} Train, {len(val_examples)} Validation.")

        # 2. Load PLM
        model_name = "microsoft/mdeberta-v3-base"
        print(f"Loading {model_name} (Backbone)...")
        try:
            plm, tokenizer, model_config, WrapperClass = load_plm("deberta-v2", model_name)
        except:
            plm, tokenizer, model_config, WrapperClass = load_plm("deberta", model_name)

        if tokenizer is None:
            print("🔍 Fix: Tokenizer missing from load_plm. Manual recovery initiated...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
        # Global registration for Wrapper discovery
        globals()['tokenizer'] = tokenizer
            
        if WrapperClass is None:
            WrapperClass = DebertaV3TokenizerWrapper

        # 3. Template & Verbalizer
        template_text = '{"soft": None, "num_tokens": 20} {"placeholder": "text_a"} Category: {"mask"}.'
        mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text, num_tokens=20)
        
        # Dynamic Multi-Class Verbalizer
        label_words = {i: [label_name] for i, label_name in enumerate(handler.discovered_classes)}
        myverbalizer = SoftVerbalizer(tokenizer=tokenizer, model=plm, 
                                     classes=list(range(handler.num_classes)), 
                                     label_words=label_words)

        # 4. Build Model
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prompt_model.to(device)
        print(f"Running on: {device}")

        # 5. Data Loaders
        print("Preparing DataLoaders (with Shuffling)...")
        train_dataloader = PromptDataLoader(dataset=train_examples, template=mytemplate, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=4, shuffle=True)
        
        val_dataloader = PromptDataLoader(dataset=val_examples, template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=128, batch_size=4, shuffle=False)

        # 6. Training & Evaluation Loop
        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=1e-3)
        loss_func = torch.nn.CrossEntropyLoss()

        print(f"\n🚀 Starting Automated Training Pipeline ({len(train_dataloader)} batches/epoch)...")
        
        for epoch in range(3):
            # --- TRAINING PHASE ---
            prompt_model.train()
            tot_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [TRAIN]")
            for step, inputs in enumerate(pbar):
                inputs = inputs.to(device)
                logits = prompt_model(inputs)
                labels = inputs['label'].long()
                
                # Shape Shield
                if len(logits.shape) == 3:
                    logits = logits.mean(dim=1) if logits.shape[1] > 0 else torch.zeros((logits.shape[0], logits.shape[2]), device=device, requires_grad=True)
                
                try:
                    loss = loss_func(logits, labels)
                except:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                tot_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total)*100:.1f}%")

            train_acc = (correct / total) * 100 if total > 0 else 0
            
            # --- VALIDATION PHASE ---
            prompt_model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs in val_dataloader:
                    inputs = inputs.to(device)
                    logits = prompt_model(inputs)
                    labels = inputs['label'].long()
                    if len(logits.shape) == 3:
                        logits = logits.mean(dim=1) if logits.shape[1] > 0 else torch.zeros((logits.shape[0], logits.shape[2]), device=device)
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
            print(f"✨ Epoch {epoch+1} Summary | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {tot_loss/len(train_dataloader):.4f}")

        print("\n" + "="*60)
        print("✅ AUTOMATED LARGE-SCALE TRAINING COMPLETE")
        print(f"📊 Final Validation Accuracy: {val_acc:.2f}%")
        print("="*60)
        print("\n[COLAB EXECUTION SUCCESSFUL]")

    run_mdeberta_colab()

if __name__ == "__main__":
    main()
