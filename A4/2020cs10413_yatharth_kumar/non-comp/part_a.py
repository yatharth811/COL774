import torch
import torch.nn as nn

class EncoderCNN(nn.Module):
	def __init__(self):
		super(EncoderCNN, self).__init__()

		self.model = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(32, 64, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 128, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(128, 256, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(256, 512, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.AdaptiveAvgPool2d((1, 1))
		)

	def forward(self, x):
		return self.model(x)


import torch
import torch.nn as nn
import csv

vocabulary = set()

def build_vocabulary(csv_file_path: str):
  global vocabulary
  with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
      if (i == 0): continue
      for tokens in row[1].split():
        vocabulary.add(tokens)
        
build_vocabulary('col_774_A4_2023/HandwrittenData/train_hw.csv')
build_vocabulary('col_774_A4_2023/HandwrittenData/val_hw.csv')
build_vocabulary('col_774_A4_2023/SyntheticData/train.csv')
build_vocabulary('col_774_A4_2023/SyntheticData/val.csv')
vocabulary.add('<start>')
vocabulary.add('<unk>')
vocabulary.add('<pad>')
vocabulary.add('<eof>')

id_to_token = {}
token_to_id = {}

for i, token in enumerate(vocabulary):
  id_to_token[i] = token
  token_to_id[token] = i

class Decoder(nn.Module):
  def __init__(self, output_vocabulary_size: int, context_size: int, embedding_dimension: int, hidden_layer_dimension: int):
    super(Decoder, self).__init__()

    self.context_size = context_size
    self.embedding = nn.Embedding(output_vocabulary_size, embedding_dimension)
    self.output_layer = nn.Linear(hidden_layer_dimension, output_vocabulary_size)
    self.lstm = nn.LSTM(embedding_dimension + context_size, hidden_layer_dimension, batch_first=True)

    # Define initial hidden and cell states as lambda functions
    self.initialize_hidden = lambda batch_size: (
        torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device),
        torch.zeros(1, batch_size, self.lstm.hidden_size, device=self.device)
    )


  def forward(self, context_vector, target_sequence, max_seq_length):
    # Initialize the hidden state and cell state
    batch_size = context_vector.size(0)
    self.device = context_vector.device

    # Initial hidden and cell states
    h_t, c_t = self.initialize_hidden(batch_size)
    
    # Embedded Sequence
    embedded_sequence = self.embedding(target_sequence) if target_sequence is not None else None

    # Initialize the output tensor
    start_token_index = token_to_id['<start>']
    outputs = [torch.zeros((batch_size, 1, len(vocabulary)), device=self.device)]
    outputs[0][:, :, start_token_index] = 1

    # Expand context vector
    expanded_context = context_vector.unsqueeze(1)
    
    embedded_sequence = embedded_sequence[:,t].unsqueeze(1) if embedded_sequence is not None else embedded_sequence

    # Forward pass through the LSTM
    for t in range(max_seq_length - 1):
      x = self.embedding(torch.argmax(outputs[-1],dim=-1).reshape(batch_size,1)) if embedded_sequence is None else embedded_sequence
      input_t = torch.cat((x, expanded_context), dim=2)
      output, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
      outputs.append(self.output_layer(output))

    return torch.stack(outputs, dim = 1)