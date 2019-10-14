import mlrose

class StructuredBits:
    
    def __init__(self):
        pass

    def evaluate(self, state):
        length = len(state)
        score = 0
        if length % 2 == 0:
            even_parity_middle = int(length / 2 - 1)
            odd_parity_middle = even_parity_middle + 1
        else:
            even_parity_middle = int(length / 2)
            odd_parity_middle = even_parity_middle
        for i in range(0, even_parity_middle):
            bit_a = state[i]
            bit_b = state[length - i - 1]
            if i % 2 == 0 and self.__parity__(bit_a, bit_b, state[even_parity_middle]) == 0:
                score += 1
            elif i % 2 == 1 and self.__parity__(bit_a, bit_b, state[odd_parity_middle]) == 1:
                score += 1
        return score
    
    def get_prob_type(self):
        return 'discrete'

    @staticmethod
    def __parity__(bit_a, bit_b, bit_c):
        return (bit_a + bit_b + bit_c) % 2
        


