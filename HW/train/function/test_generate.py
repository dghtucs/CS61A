from simple_large_model import RealTokenizer, TransformerChatbot, generate_response

# quick smoke test without training
if __name__ == '__main__':
    tokenizer = RealTokenizer()
    tokenizer.build_vocab(["Hello how are you", "I am fine", "What about you"]) 
    model = TransformerChatbot(vocab_size=1000, model_dim=128, num_heads=4, num_layers=2)
    resp = generate_response(model, tokenizer, "hello")
    print('Generated:', resp)
