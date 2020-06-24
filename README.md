# Chinese Text Classification
Based on the simplified implementation of TensorFlow on the Chinese data set. 
It uses character-level CNN to classify Chinese text and achieves good results.

## Environment
- TensorFlow 1.3 (For TensorFlow 2, using `tf.disable_v2_behavior()` to update)
- numpy
- scikit-learn
- scipy

## Dataset
Use a subset of THUCNews for training and testing, please go to THUCTC for a data set: 
[an efficient Chinese text classification toolkit to download,](http://thuctc.thunlp.org) 
please follow the open source agreement of the data provider.

This training uses 10 of them, each with 6,500 pieces of data. Categories includes: Sports, finance, real estate, home, education, technology, fashion, current affairs, games, entertainment

The data set is divided as follows:
- Training set: 5000*10 (cnews.train.txt)
- Verification set: 500*10 (cnews.val.txt)
- Test set: 1000*10 (cnews.test.txt)

## Data Preprocessing
data/cnews_loader.py is the data preprocessing file.

- read_file(): read file data;
- build_vocab(): build a vocabulary table, using character-level representation, this function will store the vocabulary table to avoid repeated processing every time;
- read_vocab(): read the vocabulary stored in the previous step and convert it to {word: id};
- read_category(): fix the category catalog and convert it to {category: id};
- to_words(): re-convert a piece of data represented by id to text;
- process_file(): Convert the data set from text to a fixed-length id sequence representation;
- batch_iter(): Prepare shuffled batch data for neural network training.

## Convolutional Neural Network
### Configuration
The configurable parameters of CNN are shown below, in cnn_model.py.

```
    embedding_dim = 64 # Word vector dimension
    seq_length = 600 # sequence length
    num_classes = 10 # number of categories
    num_filters = 128 # number of convolution kernels
    kernel_size = 5 # Convolution kernel size
    vocab_size = 5000 # Vocabulary expression is small

    hidden_dim = 128 # fully connected layer neurons

    dropout_keep_prob = 0.5 # dropout retention ratio
    learning_rate = 1e-3 # learning rate

    batch_size = 64 # training size per batch
    num_epochs = 10 # total iteration rounds

    print_per_batch = 100 # How many rounds to output the result
    save_per_batch = 10 # How many rounds are saved in tensorboard
```
### CNN Model
At first, we send our preprocessing
character data into embedding layer. The embedding processing is usually used in NLP area to transform sparse matrix to compact matrix which 
may take less space and be better for CNN model to handle. In our model, we turn 600\*5000 one hot vectors to 600\*64 word vectors by using this layer.

Then we do 1D convolution to these word vectors in CNN layer. 128 filters are selected to catch as much as possible features. 
These output will do MaxPooling next for three reasons, first is to prevent over fitting, reducing the amount of calculation 
while retaining the main features is the second and last reason is that it can reduce the number of parameters.

After catching all local features, we use fully connected layer to integrate all local features together. 
Dropout function is used to avoid over fitting, and ReLU is selected as activation function which can also solve the problem of gradient vanishing.

At last softmax regression is the method to classify our 10 categories, which can tell us that the input news belong to which category 
with highest possibility.

The training strategy we apply in this project is stochastic gradient descent. In detail, 
we set 10 epoch most for the training process, and the batch size we set is 64, that means there are about 800 iterations per epoch. 
Every 100 iterations we check and print the result by running the CNN model with validation set to see whether there is any improvement in classification. 
If there is no improvement over 1000 iterations, we auto stop our model training process in this situation.

### Training and Testing
Run python run_cnn.py train to start training. 
If you have trained before, please delete tensorboard/textcnn to avoid overlapping of TensorBoard training results for multiple times.

### Result
statistical outputs shows the accuracy at testing set is about 97%, and precision, recall and f1-score for
every category are all over 0.9. We can say that the CNN model we design is good in Chinese text classification.