# Parameters
max_input_length = 64
max_output_length = 128
input_vocab_size = 1000
output_vocab_size = 2000
embedding_size = 256
num_attention_heads = 8
embedding_size_per_head = embedding_size // num_attention_heads
num_encoders = 6
num_decoders = 6

print(f"Total number of self-attention blocks: {num_encoders + num_decoders}")
print(f"Total number of skip connections: {2 * num_encoders + 3 * num_decoders}")

print(f"Total number of elements in the output tensor of the last encoder block: "
      f"{max_input_length * embedding_size}")

print(f"(Total number of elements in the output tensor of the last decoder block: "
      f"{max_output_length * embedding_size})")

print(f"Total number of elements in the output tensor of scaled dot-product attention of an encoder-decoder block: "
      f"{max_output_length * max_input_length}")

print(f"Alternatively: Dimensions of the attention weight matrix for a single head of an encoder-decoder block: "
      f"{max_output_length * max_input_length}")

print(f"Total number of output units of the final fully connected layer: {output_vocab_size}")

print(f"Total number of weights (no biases!) in a single multi-head self-attention block: "
      f"{3 * embedding_size**2 + num_attention_heads * embedding_size + embedding_size}")
# print(f"(Alternate calculation that might not be correct: {4 * embedding_size**2})")
