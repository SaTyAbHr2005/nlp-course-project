import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
import joblib
import textstat
import numpy as np

# ========================================================
# 1. SETUP & CONFIGURATION
# ========================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSIFIER_MODEL_NAME = 'google/electra-large-discriminator'
CLASSIFIER_PT_PATH = 'electra_large_final_model.pt'
SCALER_PATH = 'scaler.pkl'

SIMPLIFIER_MODEL_NAME = 't5-base'
SIMPLIFIER_PT_PATH = 't5_simplifier_best.pt'

LABELS = {0: "Easy (Level 1)", 1: "Medium (Level 2)", 2: "Hard (Level 3)"}
LABEL_COLORS = {0: "🟢", 1: "🟡", 2: "🔴"}

# ========================================================
# 2. LOAD CLASSIFIER (Electra Hybrid)
# ========================================================
class ElectraHybrid(nn.Module):
    def __init__(self, num_labels=3, n_numeric=10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CLASSIFIER_MODEL_NAME)
        hidden_size = 1024 # ELECTRA-large hidden size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + n_numeric, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, numeric_features):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        combined = torch.cat([cls, numeric_features], dim=1)
        return self.classifier(combined)

print("Loading Classifier Tokenizer & Scaler...")
clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
scaler = joblib.load(SCALER_PATH)

print("Loading ELECTRA-large Classifier Weights...")
classifier = ElectraHybrid().to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_PT_PATH, map_location=DEVICE))
classifier.eval()

# ========================================================
# 3. LOAD SIMPLIFIER (T5)
# ========================================================
print("Loading T5 Simplifier Tokenizer & Weights...")
simp_tokenizer = T5TokenizerFast.from_pretrained(SIMPLIFIER_MODEL_NAME)
simplifier = T5ForConditionalGeneration.from_pretrained(SIMPLIFIER_MODEL_NAME).to(DEVICE)
simplifier.load_state_dict(torch.load(SIMPLIFIER_PT_PATH, map_location=DEVICE))
simplifier.eval()

print("All models loaded successfully!")

# ========================================================
# 4. HELPER FUNCTIONS
# ========================================================
def extract_features(text):
    words = text.split()
    word_count = len(words)
    if word_count == 0: return [0.0]*10
    
    sentence_count = textstat.sentence_count(text)
    if sentence_count == 0: sentence_count = 1
        
    avg_word_length = sum(len(w) for w in words) / word_count
    avg_sentence_length = word_count / sentence_count
    
    unique_words = len(set(w.lower() for w in words))
    unique_word_ratio = unique_words / word_count
    
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / word_count
    
    diff_words = textstat.difficult_words(text)
    diff_word_pct = diff_words / word_count if word_count > 0 else 0
    
    return [
        float(textstat.flesch_kincaid_grade(text)),
        float(word_count),
        float(sentence_count),
        float(avg_word_length),
        float(avg_sentence_length),
        float(textstat.syllable_count(text)),
        float(unique_word_ratio),
        float(long_word_ratio),
        float(textstat.flesch_reading_ease(text)),
        float(diff_word_pct)
    ]

def predict_difficulty(text):
    features = np.array([extract_features(text)])
    scaled_feats = scaler.transform(features)
    num_tensor = torch.tensor(scaled_feats, dtype=torch.float32).to(DEVICE)
    
    enc = clf_tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        logits = classifier(
            input_ids=enc['input_ids'].to(DEVICE),
            attention_mask=enc['attention_mask'].to(DEVICE),
            token_type_ids=enc['token_type_ids'].to(DEVICE),
            numeric_features=num_tensor
        )
        pred_idx = torch.argmax(logits, dim=1).item()
        
    return pred_idx

def simplify_text(text):
    inp_text = "simplify: " + text
    inp = simp_tokenizer(inp_text, return_tensors='pt', max_length=128, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        out = simplifier.generate(
            **inp, 
            max_new_tokens=100, 
            num_beams=4, 
            early_stopping=True,
            repetition_penalty=1.2,       # Relaxed slightly to prevent hallucinations
            length_penalty=0.8            # Encourages the AI to be concise and stop generating early
        )
        
    return simp_tokenizer.decode(out[0], skip_special_tokens=True)

# ========================================================
# 5. MAIN GRADIO PIPELINE
# ========================================================
def process_text(input_text):
    if not input_text.strip():
        return "Please enter some text.", "", ""
        
    # Step 1: Predict Difficulty
    pred_idx = predict_difficulty(input_text)
    difficulty_label = f"### Input sentence is classified into: {LABEL_COLORS[pred_idx]} {LABELS[pred_idx]}"
    
    # Step 2: Simplify if necessary
    if pred_idx == 0:
        return difficulty_label, "Text is already easy! No simplification needed.", ""
    else:
        simplified_output = simplify_text(input_text)
        
        orig_grade = textstat.flesch_kincaid_grade(input_text)
        new_grade = textstat.flesch_kincaid_grade(simplified_output)
        stats = f"📉 **Reading Grade dropped from {orig_grade:.1f} to {new_grade:.1f}!**"
        
        return difficulty_label, simplified_output, stats

# ========================================================
# 6. GRADIO UI LAYOUT
# ========================================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Smart Reading Assistant")
    
    input_box = gr.Textbox(lines=5, label="Enter your text here", placeholder="Paste a paragraph...")
    analyze_btn = gr.Button("Analyze & Simplify", variant="primary")
    
    gr.Markdown("---")
    
    difficulty_output = gr.Markdown("### Input sentence is classified into: ")
    simplified_output = gr.Textbox(lines=4, label="After Simplifying, the sentence is:", interactive=False)
    stats_output = gr.Markdown("")
            
    analyze_btn.click(
        fn=process_text,
        inputs=input_box,
        outputs=[difficulty_output, simplified_output, stats_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
