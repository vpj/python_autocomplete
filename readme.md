### ⭐️ We rewrote a simpler version of this at [lab-ml/source_code_modelling](https://github.com/lab-ml/source_code_modelling) and we intend to maintain it for a while

[This](https://github.com/vpj/python_autocomplete) a toy project we started
to see how well a simple LSTM model can autocomplete python code.

It gives quite decent results by saving above 30% key strokes in most files,
and close to 50% in some.
We calculated key strokes saved by making a single (best)
prediction and selecting it with a single key.

We do a beam search to find predictions, upto ~10 characters ahead.
So far it's too inefficient, if you are wondering about editor integration.

We train and predict on after cleaning comments, strings
and blank lines in python code.
The model is trained after tokenizing python code.
It seems more efficient than character level prediction with byte-pair encoding.

A saved model is included in this repo.
It is trained on [tensorflow/models](https://github.com/tensorflow/models).

Here's a sample evaluation on a source file from validation set.
Red characters are when a auto-completion started;
i.e. user presses TAB to select the completion. 
The green character and and the following characters highlighted in gray
are auto-completed. As you can see, it starts and ends completions arbitrarily.
That is a suggestion could be 'tensorfl' and not the complete identifier
'tensorflow' which can be a little annoying in a real usage scenario.
We can limit them to finish on end of tokens to fix that.
Also you can notice that it completes across operators as well.
Increasing the length of the beam search will let it complete longer pieces of code.

<p align="center">
  <img src="/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>

## Try it yourself

1. Clone this repo

2. Install requirements from `requirements.txt`

3. Copy data to `./data/source`

4. Run `extract_code.py` to collect all python files, encode and merge them into `all.py`

5. Run `evaluate.py` to evaluate the model. I have included a checkpoint in the repo.

6. Run `train.py` to train the model
