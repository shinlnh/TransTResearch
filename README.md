# GPT BERT Theory - Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án training GPT (Generative Pre-trained Transformer) từ đầu, tập trung vào việc hiểu sâu về kiến trúc Transformer và quá trình pre-training language model.

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Tính năng](#-tính-năng)
- [Yêu cầu](#-yêu-cầu)
- [Cài đặt](#-cài-đặt)
- [Pipeline sử dụng](#-pipeline-sử-dụng)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Text Generation](#-text-generation)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Configuration](#-configuration)
- [Kết quả](#-kết-quả)

## 🎯 Tổng quan

Dự án này implement GPT architecture với các đặc điểm:

- **Kiến trúc**: Transformer decoder-only với causal self-attention
- **Scale linh hoạt**: Có thể config số layers, heads, embedding dimensions
- **Training tối ưu**: Hỗ trợ mixed precision (AMP), gradient clipping, checkpointing
- **Data processing**: Pipeline hoàn chỉnh từ raw text → tokenization → binary sequences
- **Generation**: Hỗ trợ nhiều sampling strategies (top-k, top-p, temperature, repetition penalty)

## 📁 Cấu trúc dự án

```
GPTBERT/
│
├── data/
│   ├── Regex/                          # Regex theory và examples
│   │   ├── B1.py, B2.py, B3.py        # Bài tập regex
│   │   └── TheoryRegex.md             # Lý thuyết regex
│   │
│   ├── Token/                          # Tokenization
│   │   ├── data/                       # Corpus để train tokenizer
│   │   │   └── corpus_cleaned.txt     # Cleaned text corpus (1GB+)
│   │   ├── tokenizers_bpe/            # BPE tokenizer output
│   │   │   ├── vocab.json             # Vocabulary (30k tokens)
│   │   │   ├── merges.txt             # BPE merge operations
│   │   │   └── tokenizer.json         # Tokenizer config
│   │   ├── tokenizers_wordpiece/       # WordPiece tokenizer (optional)
│   │   └── vocab/                      # Scripts to build tokenizers
│   │       ├── build_vocab.py         # Train BPE tokenizer
│   │       ├── build_wordpiece_vocab.py
│   │       ├── check_jsonvocab.py     # Verify vocab
│   │       └── clean_data.py          # Preprocess corpus
│   │
│   └── TokenizeCorpus/
│       ├── data_tokGPT/               # Tokenized corpus (train_ids.bin)
│       └── data_tokBERT/              # For BERT (if implemented)
│
├── gpt/                                # Main GPT implementation
│   ├── model/
│   │   ├── __init__.py
│   │   └── gpt.py                     # GPT model architecture
│   │       ├── GPTConfig              # Model configuration
│   │       ├── CausalSelfAttention    # Masked multi-head attention
│   │       ├── MLP                    # Feed-forward network
│   │       ├── Block                  # Transformer block
│   │       └── GPT                    # Main model class
│   │
│   ├── dataloader/
│   │   └── dataloader.py              # Efficient binary data loading
│   │       ├── GPTBinDataset          # Memmap-based dataset
│   │       └── build_dataloader       # DataLoader builder
│   │
│   ├── CutSequences/
│   │   ├── cutsequence.py             # Chop tokenized data into sequences
│   │   ├── gpt_meta.json              # Metadata (seq_len, num_samples)
│   │   ├── gpt_x.bin                  # Input sequences (2.5M samples)
│   │   └── gpt_y.bin                  # Target sequences (shifted)
│   │
│   ├── train.py                        # Training script
│   ├── eval.py                         # Evaluation script (perplexity)
│   ├── test.py                         # Text generation/inference
│   └── checkpoints/                    # Model checkpoints
│       ├── ckpt_last.pt               # Latest checkpoint
│       └── ckpt_step_*.pt             # Periodic checkpoints
│
├── venv312/                            # Python virtual environment
├── .gitignore                          # Git ignore patterns
└── README.md                           # This file
```

## ✨ Tính năng

### Model Features
- ✅ **Causal Self-Attention** với mask tam giác (autoregressive)
- ✅ **Position Embeddings** (learned positional encoding)
- ✅ **Layer Normalization** (pre-norm style)
- ✅ **Dropout** cho regularization
- ✅ **Weight Tying** (share weights giữa token embedding và LM head)
- ✅ **GELU Activation** trong MLP

### Training Features
- ✅ **Mixed Precision Training** (AMP) - giảm memory, tăng tốc
- ✅ **Gradient Clipping** - stable training
- ✅ **AdamW Optimizer** với weight decay
- ✅ **Checkpointing** - save/resume training
- ✅ **Progress Tracking** với tqdm
- ✅ **Logging** (loss, learning rate, throughput)

### Data Processing
- ✅ **BPE Tokenization** (Byte-Pair Encoding) - 30k vocab
- ✅ **Binary Data Format** - efficient storage và loading
- ✅ **Memory Mapping** - không load toàn bộ data vào RAM
- ✅ **Sliding Window** - tạo samples từ long sequences

### Text Generation
- ✅ **Temperature Sampling** - control randomness
- ✅ **Top-K Sampling** - chỉ sample từ k tokens có prob cao nhất
- ✅ **Top-P (Nucleus) Sampling** - dynamic vocabulary cutoff
- ✅ **Repetition Penalty** - giảm lặp từ
- ✅ **No-Repeat N-gram** - ngăn lặp n-gram
- ✅ **Early Stopping** với EOS token

## 🔧 Yêu cầu

### Software Requirements
```txt
Python >= 3.8
PyTorch >= 2.0
tokenizers >= 0.13.0 (HuggingFace)
numpy >= 1.21.0
tqdm >= 4.62.0 (optional, for progress bars)
```

### Hardware Requirements

**Minimum** (cho experimenting):
- RAM: 8GB
- GPU: 4GB VRAM (GTX 1650 trở lên)
- Storage: 5GB (corpus + tokenized data + checkpoints)

**Recommended** (cho full training):
- RAM: 16GB+
- GPU: 8GB+ VRAM (RTX 3070, A4000 trở lên)
- Storage: 10GB+

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/shinlnh/NLP_GPT_THEORY.git
cd NLP_GPT_THEORY
```

### 2. Tạo virtual environment
```bash
python -m venv venv312
# Windows
venv312\Scripts\activate
# Linux/Mac
source venv312/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tokenizers numpy tqdm
```

## 🚀 Pipeline sử dụng

### Bước 1: Chuẩn bị Corpus

**Option A**: Sử dụng corpus có sẵn
- Đặt file text vào `data/Token/data/corpus_cleaned.txt`
- Recommend: Wikipedia dumps, BookCorpus, C4, OpenWebText

**Option B**: Download Wikipedia dump
```bash
# Download English Wikipedia (cần ~20GB trống)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

### Bước 2: Train Tokenizer

```bash
cd data/Token/vocab
python build_vocab.py
```

**Output**:
- `data/Token/tokenizers_bpe/vocab.json` - 30,000 tokens
- `data/Token/tokenizers_bpe/merges.txt` - BPE merge rules
- `data/Token/tokenizers_bpe/tokenizer.json` - Full tokenizer config

**Verify tokenizer**:
```bash
python check_jsonvocab.py
```

### Bước 3: Tokenize Corpus

Tạo script tokenize toàn bộ corpus thành binary format:

```python
# data/TokenizeCorpus/tokenize_gpt.py
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("data/Token/tokenizers_bpe/tokenizer.json")

# Load corpus
corpus_path = Path("data/Token/data/corpus_cleaned.txt")
text = corpus_path.read_text(encoding="utf-8")

# Encode
output = tokenizer.encode(text)
ids = np.array(output.ids, dtype=np.uint32)

# Save
out_path = Path("data/TokenizeCorpus/data_tokGPT/train_ids.bin")
out_path.parent.mkdir(parents=True, exist_ok=True)
ids.tofile(out_path)
print(f"Saved {len(ids):,} tokens to {out_path}")
```

### Bước 4: Cut Sequences

Chia tokenized data thành fixed-length sequences:

```bash
cd gpt/CutSequences
python cutsequence.py
```

**Config trong script**:
- `SEQ_LEN = 512` - context window
- `STRIDE = 512` - sliding window step (512 = no overlap)
- `DTYPE = np.uint32` - data type

**Output**:
- `gpt_x.bin` - input sequences [N, 512]
- `gpt_y.bin` - target sequences (shifted right)
- `gpt_meta.json` - metadata

### Bước 5: Train Model

```bash
cd gpt
python train.py \
  --batch_size 8 \
  --epochs 10 \
  --lr 3e-4 \
  --n_layers 6 \
  --n_heads 6 \
  --n_embd 384 \
  --dropout 0.1 \
  --amp \
  --log_every 100 \
  --save_every 1000 \
  --out_dir checkpoints
```

**Main arguments**:
- `--batch_size`: Số samples/batch (giảm nếu OOM)
- `--epochs`: Số epochs training
- `--lr`: Learning rate (3e-4 cho GPT small)
- `--n_layers`: Số transformer blocks (6-12 cho small models)
- `--n_heads`: Số attention heads (6-12)
- `--n_embd`: Embedding dimension (384, 768, 1024...)
- `--dropout`: Dropout rate (0.1 là standard)
- `--amp`: Enable mixed precision (recommend)
- `--grad_clip`: Gradient clipping threshold (1.0)
- `--save_every`: Save checkpoint mỗi N steps
- `--resume`: Path to checkpoint để resume training

### Bước 6: Evaluate Model

```bash
python eval.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --batch_size 8 \
  --max_batches 0
```

**Output**: Validation loss và perplexity

### Bước 7: Generate Text

```bash
python test.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 50 \
  --repetition_penalty 1.1
```

**Generation parameters**:
- `--prompt`: Input text (hoặc dùng `--prompt_ids` cho comma-separated IDs)
- `--max_new_tokens`: Số tokens sinh ra
- `--temperature`: Sampling temperature (0.8-1.0 cho creative, <0.8 cho conservative)
- `--top_k`: Top-k sampling (40-100)
- `--top_p`: Nucleus sampling (0.9-0.95)
- `--repetition_penalty`: Penalty cho repeated tokens (>1.0)
- `--no_repeat_ngram_size`: Block n-gram repeats (0 = off, 3-4 is good)
- `--eos_token_id`: EOS token để stop generation

## 🎓 Training

### Training Script Details

File [`train.py`](gpt/train.py) có đầy đủ options:

```python
@dataclass
class TrainConfig:
    # Data paths
    meta_path: str = "gpt/CutSequences/gpt_meta.json"
    vocab_json: str = "data/Token/tokenizers_bpe/vocab.json"
    
    # Model architecture
    n_layers: int = 6        # Transformer blocks
    n_heads: int = 6         # Attention heads per block
    n_embd: int = 384        # Embedding dimension
    dropout: float = 0.1     # Dropout rate
    
    # Training hyperparameters
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9       # Adam beta1
    beta2: float = 0.95      # Adam beta2
    grad_clip: float = 1.0
    
    # System
    device: str = "cuda"     # or "cpu"
    amp: bool = True         # Mixed precision
    num_workers: int = 0     # DataLoader workers
    
    # Logging/Checkpointing
    log_every: int = 100
    save_every: int = 1000
    out_dir: str = "gpt/checkpoints"
    resume: str = ""         # Path to resume from
```

### Training Tips

1. **Memory optimization**:
   - Giảm `batch_size` nếu OOM
   - Enable `amp=True` (mixed precision)
   - Giảm `seq_len` nếu cần (512 → 256)

2. **Training stability**:
   - Gradient clipping (`grad_clip=1.0`)
   - Learning rate warmup (có thể implement)
   - Monitor loss - nếu diverge thì giảm LR

3. **Speed optimization**:
   - GPU với Tensor Cores (RTX series)
   - `pin_memory=True` trong DataLoader
   - Tăng `num_workers` nếu CPU bottleneck

## 📊 Evaluation

File [`eval.py`](gpt/eval.py) tính validation metrics:

```bash
python eval.py --checkpoint checkpoints/ckpt_last.pt
```

**Metrics**:
- **Loss**: Cross-entropy loss
- **Perplexity**: $\text{PPL} = e^{\text{loss}}$ (lower is better)

Good perplexity targets:
- ~100-200: Decent small model
- ~50-100: Good model
- <50: Very good (GPT-2 level: ~30-40)

## 🎨 Text Generation

File [`test.py`](gpt/test.py) hỗ trợ nhiều sampling strategies:

### Basic Generation
```bash
python test.py --prompt "Hello world"
```

### Creative Generation (high diversity)
```bash
python test.py \
  --prompt "In a galaxy far away" \
  --temperature 1.0 \
  --top_p 0.95 \
  --repetition_penalty 1.2
```

### Conservative Generation (focused)
```bash
python test.py \
  --prompt "The capital of France is" \
  --temperature 0.5 \
  --top_k 10
```

### Sampling Strategies Explained

1. **Temperature**: Scale logits before softmax
   - `T > 1.0`: More random (creative)
   - `T < 1.0`: More focused (conservative)
   - `T = 0.0`: Greedy (always pick max)

2. **Top-K**: Chỉ sample từ K tokens có prob cao nhất
   - `k=1`: Greedy
   - `k=50`: Standard
   - `k=100`: More diverse

3. **Top-P (Nucleus)**: Sample từ smallest set có cumulative prob ≥ p
   - `p=0.9`: Safe
   - `p=0.95`: Standard (GPT-2/3 default)
   - `p=1.0`: No filtering

4. **Repetition Penalty**: Penalize tokens đã xuất hiện
   - `penalty=1.0`: No penalty
   - `penalty=1.1-1.2`: Mild (recommend)
   - `penalty>1.5`: Aggressive

## 🏗️ Kiến trúc mô hình

### GPT Architecture

```
Input Text → Tokenization → Token IDs
                               ↓
                    Token Embedding (vocab_size → n_embd)
                               ↓
                    Position Embedding (0..511 → n_embd)
                               ↓
                            Dropout
                               ↓
        ┌──────────────────────┴───────────────────┐
        │     Transformer Block x N_LAYERS          │
        │  ┌──────────────────────────────────┐    │
        │  │  LayerNorm                       │    │
        │  │  Causal Self-Attention (masked)  │    │
        │  │  Residual Connection             │    │
        │  └──────────────────────────────────┘    │
        │  ┌──────────────────────────────────┐    │
        │  │  LayerNorm                       │    │
        │  │  MLP (4x expansion + GELU)       │    │
        │  │  Residual Connection             │    │
        │  └──────────────────────────────────┘    │
        └───────────────────┬──────────────────────┘
                            ↓
                      LayerNorm
                            ↓
                 LM Head (n_embd → vocab_size)
                            ↓
                      Logits / Loss
```

### Key Components

**1. CausalSelfAttention**
```python
class CausalSelfAttention(nn.Module):
    # Scaled Dot-Product Attention với causal mask
    # Q, K, V = Linear(x)
    # Attention = softmax(QK^T / √d_k) V
    # Mask tam giác: positions chỉ attend vào past
```

**2. MLP (Feed-Forward)**
```python
class MLP(nn.Module):
    # 2-layer MLP với GELU
    # Hidden = 4 * n_embd (expansion ratio)
    # x → Linear(4x) → GELU → Linear(x) → Dropout
```

**3. Transformer Block**
```python
class Block(nn.Module):
    # Pre-Norm architecture
    # x = x + Attention(LayerNorm(x))
    # x = x + MLP(LayerNorm(x))
```

### Model Sizes

| Config | Layers | Heads | Embd | Params | VRAM |
|--------|--------|-------|------|---------|------|
| Tiny   | 4      | 4     | 256  | ~10M    | 2GB  |
| Small  | 6      | 6     | 384  | ~25M    | 4GB  |
| Medium | 12     | 12    | 768  | ~117M   | 8GB  |
| Large  | 24     | 16    | 1024 | ~345M   | 16GB |

*Note*: VRAM estimate cho batch_size=8, seq_len=512 với mixed precision

## ⚙️ Configuration

### Model Config ([`gpt.py`](gpt/model/gpt.py))

```python
@dataclass
class GPTConfig:
    vocab_size: int          # Vocabulary size (30k cho BPE)
    block_size: int          # Context length (512)
    n_layers: int = 6        # Số transformer blocks
    n_heads: int = 6         # Attention heads
    n_embd: int = 384        # Embedding dimension
    dropout: float = 0.1     # Dropout probability
    bias: bool = True        # Bias trong Linear layers
```

### Training Config ([`train.py`](gpt/train.py))

Xem section [Training](#-training) phía trên.

### DataLoader Config ([`dataloader.py`](gpt/dataloader/dataloader.py))

```python
@dataclass
class LoaderConfig:
    x_bin: str
    y_bin: str
    seq_len: int
    dtype: str = "uint32"
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "cuda"
```

## 📈 Kết quả

### Training Progress

Với config mặc định (6 layers, 6 heads, 384 embd) trên 2.5M samples (512 seq_len):

| Step | Loss | Perplexity | Time |
|------|------|------------|------|
| 0    | ~10  | ~22000     | -    |
| 1k   | ~6   | ~403       | 1h   |
| 10k  | ~4   | ~55        | 8h   |
| 100k | ~3   | ~20        | 3d   |
| 1M   | ~2.5 | ~12        | 2w   |

*Hardware*: RTX 3070, batch_size=8, AMP enabled

### Sample Generations

**Prompt**: "Once upon a time"

**Temperature=0.8**:
```
Once upon a time, there was a young man who lived in the countryside.
He was very poor and had no family. One day, he decided to go to the
city to find work...
```

**Temperature=1.2** (more creative):
```
Once upon a time in ancient lands, mystical creatures roamed the forests
and mountains. Dragons soared through cloudy skies while wizards practiced
their magical arts...
```

## 🔬 So sánh với GPT-2

| Feature | This Project | GPT-2 Small |
|---------|--------------|-------------|
| Layers | 6 (default) | 12 |
| Heads | 6 (default) | 12 |
| d_model | 384 (default) | 768 |
| Context | 512 | 1024 |
| Params | ~25M | 117M |
| Vocab | 30k BPE | 50k BPE |
| Training | ~1-2M steps | ~1M steps |

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Attention Is All You Need** (Vaswani et al., 2017) - Original Transformer paper
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2 paper
- **HuggingFace Tokenizers** - Fast tokenization library
- **PyTorch** - Deep learning framework

## 📧 Contact

- GitHub: [@shinlnh](https://github.com/shinlnh)
- Repository: [NLP_GPT_THEORY](https://github.com/shinlnh/NLP_GPT_THEORY)

## 🔗 Resources

- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Note**: Đây là educational project tập trung vào hiểu sâu về GPT architecture. Để sử dụng production-ready models, nên dùng HuggingFace Transformers với pre-trained checkpoints.
