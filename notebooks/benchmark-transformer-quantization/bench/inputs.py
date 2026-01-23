import torch
import numpy as np

class TransformerInputGenerator:
    """
    Generates deterministic, semantically valid inputs to prevent
    KV-cache crashes (e.g., random beam_idx issues).
    """
    def __init__(self, tokenizer, batch_size=1, seq_len=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_inputs(self):
        """
        Returns a dictionary of inputs.
        CRITICAL: beam_idx is initialized to zeros for greedy search.
        Random beam_idx causes 'ScaledDotProductAttention' crashes.
        """
 
        vocab_size = self.tokenizer.vocab_size
        input_ids = torch.arange(0, self.seq_len).repeat(self.batch_size, 1) % vocab_size
        
        # 2. Valid Attention Mask (No padding for benchmarking)
        attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.long)
        
        # 3. Position IDs (Strict sequential ordering)
        position_ids = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).repeat(self.batch_size, 1)

        # 4. Beam Index 
        # Must strictly be valid indices [0, num_beams-1].
        # For standard benchmarking, we assume greedy/single beam -> All Zeros.
        beam_idx = torch.zeros((self.batch_size, self.seq_len), dtype=torch.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": beam_idx
        }