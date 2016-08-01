tflearn_seq2seq
===============

Pedagogical example of sequence to sequence recurrent neural network
with [TensorFlow](https://www.tensorflow.org/) and
[TFLearn](http://tflearn.org/).

This code provides a complete, pegagogical, working example of a
seq2seq RNN, implemented using [TFLearn](http://tflearn.org/), which
transforms input sequences of integers, to output sequences of
integers.  

The desired transformation is defined by a pattern-generating python
function, in the file
[pattern.py](https://github.com/ichuang/tflearn_seq2seq/blob/master/pattern.py).
Examples are provided showing simple patterns, including sorting and
reversing the input sequence.  The lengths of the input and output
sequences can be set, and other patterns can be added.

Tensorflow's
[seq2seq.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py)
library is used for the RNN.  "embedding_rnn" is used by default, but
"embedding_attention" works well (often better).  See the [seq2seq
tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html)
provided by TensorFlow for an overview.  Overall, the structure is
like this (image from https://github.com/farizrahman4u/seq2seq):

![seq2seq](http://i64.tinypic.com/30136te.png)

The basic idea is that the integer sequence values are converted,
using an embedding map, to a vector in a high-dimensional space.  RNN
cells then take vectors in this space as input.  After the input
sequence ends, the first output cell is fed a special "GO" symbol,
which triggers the start of an output sequence.  When making
predictions, the output cells each take as input, the output of the
previous cell.  When training, the inputs are taken from the previous
expected symbol (embedded in the output's high dimensional embedding
space).  

Basic Usage
===========

    usage: tflearn_seq2seq.py [-h] [-v] [-m MODEL] [-r LEARNING_RATE] [-e EPOCHS]
                              [-i INPUT_WEIGHTS] [-o OUTPUT_WEIGHTS]
                              [-p PATTERN_NAME] [-n NAME] [--in-len IN_LEN]
                              [--out-len OUT_LEN] [--from-file FROM_FILE]
                              [--iter-num ITER_NUM] [--data-dir DATA_DIR]
                              [-L NUM_LAYERS] [--cell-size CELL_SIZE]
                              [--cell-type CELL_TYPE]
                              [--embedding-size EMBEDDING_SIZE]
                              [--tensorboard-verbose TENSORBOARD_VERBOSE]
                              cmd [cmd_input [cmd_input ...]]
    
    usage: %prog [command] [args...] ...
    
    Commands:
    
    train - give size of training set to use, as argument
    predict - give input sequence as argument (or specify inputs via --from-file <filename>)
    
    positional arguments:
      cmd                   command
      cmd_input             input to command
    
    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase output verbosity (add more -v to increase versbosity)
      -m MODEL, --model MODEL
                            seq2seq model name: either embedding_rnn (default) or embedding_attention
      -r LEARNING_RATE, --learning-rate LEARNING_RATE
                            learning rate (default 0.0001)
      -e EPOCHS, --epochs EPOCHS
                            number of trainig epochs
      -i INPUT_WEIGHTS, --input-weights INPUT_WEIGHTS
                            tflearn file with network weights to load
      -o OUTPUT_WEIGHTS, --output-weights OUTPUT_WEIGHTS
                            new tflearn file where network weights are to be saved
      -p PATTERN_NAME, --pattern-name PATTERN_NAME
                            name of pattern to use for sequence
      -n NAME, --name NAME  name of model, used when generating default weights filenames
      --in-len IN_LEN       input sequence length (default 10)
      --out-len OUT_LEN     output sequence length (default 10)
      --from-file FROM_FILE
                            name of file to take input data sequences from (json format)
      --iter-num ITER_NUM   training iteration number; specify instead of input- or output-weights to use generated filenames
      --data-dir DATA_DIR   directory to use for storing checkpoints (also used when generating default weights filenames)
      -L NUM_LAYERS, --num-layers NUM_LAYERS
                            number of RNN layers to use in the model (default 1)
      --cell-size CELL_SIZE
                            size of RNN cell to use (default 32)
      --cell-type CELL_TYPE
                            type of RNN cell to use (default BasicLSTMCell)
      --embedding-size EMBEDDING_SIZE
                            size of embedding to use (default 20)
      --tensorboard-verbose TENSORBOARD_VERBOSE
                            tensorboard verbosity level (default 0)

Training
========

This command trains an embedding_attention seq2seq RNN on 100,000
input sequences, using the "reversed" pattern (for which the output
sequence is the reverse of the input sequence), using 10 epochs:

    python tflearn_seq2seq.py -v -v -o weights.tfl -p reversed -m embedding_attention -e 10 train 100000

Note that 10% of the training dataset is set aside for validation.
The output should be something like this:

    [TFLearnSeq2Seq] Training on 100000 point dataset (pattern 'reversed'), with 10 epochs
      model parameters: {
        "cell_size": 32,
        "cell_type": "BasicLSTMCell",
        "embedding_size": 20,
        "num_layers": 1,
        "tensorboard_verbose": 0,
        "learning_rate": 0.0001
    }
    ---------------------------------
    Run id: TFLearnSeq2Seq
    Log directory: /tmp/tflearn_logs/
    ---------------------------------
    Training samples: 90000
    Validation samples: 10000
    --
    Training Step: 5000  | total loss: 0.01601
    | Adam | epoch: 007 | loss: 0.01601 - acc: 0.9997 | val_loss: 0.01598 - val_acc: 0.9998 -- iter: 09216/90000

    Done!
    Saved weights.tfl

In this example, final weights are saved to the file `weights.tfl` (and `weights.tfl.meta`).  

Predicting
==========

This command generates the sequence predicted by the RNN, for the
given input `0 1 2 3 4 5 6 7 8 9`, using weights from the file `weights.tfl`:

    python tflearn_seq2seq.py -i weights.tfl -v -p reversed -m embedding_attention predict 0 1 2 3 4 5 6 7 8 9

The output should be something like this:

    ==> For input [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], prediction=[8 8 5 3 6 5 4 2 3 1] (expected=[9 8 7 6 5 4 3 2 1 0])

Note how the prediction (`[8 8 5 3 6 5 4 2 3 1]`) is close to, but not exactly, what was ideally expected (`[9 8 7 6 5 4 3 2 1 0]`).

Here's another example, using a set of pre-trained weights for the "sorted" sequence pattern:

    python tflearn_seq2seq.py -i TRAINED_WEIGHTS/ts2s__attention__sorted_1.tfl \
           -p sorted --in-len=20 --out-len=20 -m embedding_attention  \
           predict 9 8 7 6 5 4 3 2 1 2 5 1 8 7 7 3 9 1 4 6

The output gives:

    [TFLearnSeq2Seq] model weights loaded from t2s__basic__sorted_1.tfl
    ==> For input [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 5, 1, 8, 7, 7, 3, 9, 1, 4, 6], prediction=[1 1 1 2 2 3 3 4 4 5 5 6 6 7 7 7 7 8 8 8] (expected=[1 1 1 2 2 3 3 4 4 5 5 6 6 7 7 7 8 8 9 9])

which, again, is close (but not exact).

Better results could probably be obtained by using a more complex model, e.g. with larger LSTM cells, or more layers, or more training.

Testing
=======

Unit tests are provided, implemented using [pytest](http://doc.pytest.org/en/latest/).  Run these using:

    py.test tflearn_seq2seq.py

