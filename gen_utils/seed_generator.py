import numpy as np 
from typing import List

def generate_copy_seed_sequence(seed_length: int, training_data: List[List[float]]) -> List[float]:

    num_examples = training_data.shape[0]
    example_len = training_data.shape[1]

    randIdx = np.random.randint(num_examples, size=1)[0]
    randSeed = np.concatenate(tuple([training_data[randIdx + i] for i in range(seed_length)]), axis=0)
    seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))

    return seedSeq