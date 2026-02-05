"""simple_large_model.py

实现一个简单的对话模型（基于 Transformer 编码器-解码器架构）。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class RealTokenizer:
    """真实的分词器，使用 NLTK 实现。"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token_id = None

    def build_vocab(self, sentences):
        """根据句子构建词汇表。"""
        vocab = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            vocab.update(tokens)
        vocab = sorted(vocab)
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.word2idx[self.eos_token] = len(self.word2idx)
        self.word2idx[self.pad_token] = len(self.word2idx)
        self.word2idx[self.unk_token] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        # 更新特殊 token 的 id
        self.eos_token_id = self.word2idx[self.eos_token]

    def encode(self, text):
        """将文本转换为 token 序列。"""
        tokens = word_tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens] + [self.word2idx[self.eos_token]]

    def decode(self, tokens):
        """将 token 序列转换回文本。"""
        words = [self.idx2word.get(token, self.unk_token) for token in tokens if token != self.word2idx[self.eos_token]]
        return " ".join(words)


class DialogueDataset(Dataset):
    """对话数据集，用于训练和测试。"""
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TransformerChatbot(nn.Module):
    """简单的 Transformer 编码器-解码器架构。"""
    def __init__(self, vocab_size, model_dim, num_heads, num_layers):
        super(TransformerChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src).transpose(0, 1)  # 调整为 [seq_len, batch_size, embedding_dim]
        tgt = self.embedding(tgt).transpose(0, 1)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = output.transpose(0, 1)  # 调整回 [batch_size, seq_len, embedding_dim]
        return self.fc(output)


def train_chatbot():
    """训练对话模型。"""
    # 超参数
    vocab_size = 1000
    model_dim = 128
    num_heads = 4
    num_layers = 2
    batch_size = 32
    epochs = 5

    # 数据集（示例数据）
    inputs = torch.randint(0, vocab_size, (1000, 10))  # 用户输入
    targets = torch.randint(0, vocab_size, (1000, 10))  # 模型回复
    dataset = DialogueDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = TransformerChatbot(vocab_size, model_dim, num_heads, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            tgt_input = tgt[:, :-1]  # 去掉最后一个词
            tgt_output = tgt[:, 1:]  # 去掉第一个词
            outputs = model(src, tgt_input)
            loss = criterion(outputs.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("对话模型训练完成！")


def save_model(model, path):
    """保存模型权重。"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


def load_model(model_class, path, *args):
    """加载模型权重。"""
    model = model_class(*args)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"模型已从 {path} 加载")
    return model


def generate_response(model, tokenizer, input_text, max_length=20):
    """根据用户输入生成回复。"""
    model.eval()
    # 支持 HuggingFace tokenizer (AutoTokenizer) 或自定义 RealTokenizer
    if AutoTokenizer is not None and hasattr(tokenizer, 'vocab_size') and hasattr(tokenizer, 'decode'):
        # HuggingFace tokenizer 路径
        input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        # 使用 eos_token_id 作为起始或结束标识
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id is None:
            # fallback to 0
            eos_id = 0
        generated_tokens = [eos_id]
        for _ in range(max_length):
            with torch.no_grad():
                tgt_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)
                output = model(input_tensor, tgt_tensor)
                # 输出形状: [batch_size, seq_len, vocab]
                next_token_logits = output[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs).item()
                if next_token == eos_id:
                    break
                generated_tokens.append(next_token)
        # HF tokenizer.decode 支持 id 列表
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        # RealTokenizer 路径（原有实现）
        input_tokens = tokenizer.encode(input_text)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

        generated_tokens = [tokenizer.eos_token_id]
        for _ in range(max_length):
            with torch.no_grad():
                tgt_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)
                output = model(input_tensor, tgt_tensor)
                next_token_logits = output[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs).item()
                if next_token == tokenizer.eos_token_id:
                    break
                generated_tokens.append(next_token)

        return tokenizer.decode(generated_tokens)


def interactive_chat(model, tokenizer):
    """交互式对话。"""
    print("开始对话（输入 'exit' 退出）")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("对话结束！")
            break
        response = generate_response(model, tokenizer, user_input)
        print(f"机器人: {response}")


if __name__ == "__main__":
    # 训练模型
    train_chatbot()
    # 初始化 tokenizer（优先使用 Hugging Face 的 AutoTokenizer）
    hf_tok = None
    if AutoTokenizer is not None:
        try:
            hf_tok = AutoTokenizer.from_pretrained("gpt2")
            if hf_tok.pad_token is None:
                hf_tok.add_special_tokens({"pad_token": "<pad>"})
        except Exception as e:
            print("Warning: failed to load HF tokenizer:", e)
            hf_tok = None

    if hf_tok is not None:
        tokenizer = hf_tok
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = RealTokenizer()
        tokenizer.build_vocab(["Hello, how are you?", "I am fine, thank you.", "What about you?"])
        vocab_size = len(tokenizer.word2idx)

    # 保存和加载模型
    model_path = "chatbot_model.pth"
    model = TransformerChatbot(vocab_size, 128, 4, 2)
    save_model(model, model_path)
    loaded_model = load_model(TransformerChatbot, model_path, vocab_size, 128, 4, 2)

    # 自动启动交互式对话
    interactive_chat(loaded_model, tokenizer)