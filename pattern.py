'''
Define pattern of integers for seq2seq example.
'''
import numpy as np

class SequencePattern(object):

    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    INPUT_MAX_INT = 9
    OUTPUT_MAX_INT = 9
    PATTERN_NAME = "sorted"

    def __init__(self, name=None, in_seq_len=None, out_seq_len=None):
        if name is not None:
            assert hasattr(self, "%s_sequence" % name)
            self.PATTERN_NAME = name
        if in_seq_len:
            self.INPUT_SEQUENCE_LENGTH = in_seq_len
        if out_seq_len:
            self.OUTPUT_SEQUENCE_LENGTH = out_seq_len

    def generate_output_sequence(self, x):
        '''
        For a given input sequence, generate the output sequence.  x is a 1D numpy array 
        of integers, with length INPUT_SEQUENCE_LENGTH.
        
        Returns a 1D numpy array of length OUTPUT_SEQUENCE_LENGTH
        
        This procedure defines the pattern which the seq2seq RNN will be trained to find.
        '''
        return getattr(self, "%s_sequence" % self.PATTERN_NAME)(x)

    def maxmin_dup_sequence(self, x):
        '''
        Generate sequence with [max, min, rest of original entries]
        '''
        x = np.array(x)
        y = [ x.max(), x.min()] +  list(x[2:])
        return np.array(y)[:self.OUTPUT_SEQUENCE_LENGTH]	# truncate at out seq len

    def sorted_sequence(self, x):
        '''
        Generate sorted version of original sequence
        '''
        return np.array( sorted(x) )[:self.OUTPUT_SEQUENCE_LENGTH]

    def reversed_sequence(self, x):
        '''
        Generate reversed version of original sequence
        '''
        return np.array( x[::-1] )[:self.OUTPUT_SEQUENCE_LENGTH]

