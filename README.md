# SVMs for Alzheimer detection

This is an implementation of an SVM classifier for Alzheimer detection, given a set of blog posts written by patients and family members. It will give us the opportunity to have a look at the output of an SVM, and also to understand how to be conifident about our results (spoiler... the main question will be: are our features sensible?)

### The data

The data is provided to you in a processed form, in the *data* directory. The *class1* folder contains data from Alzheimer patients, while the *class2* folder contains data from the control group. 

If you want to know where the data came from, note that the raw blog posts can be extracted using the script here: [https://github.com/vmasrani/blog_corpus](https://github.com/vmasrani/blog_corpus). This dataset was produced for the following publication:

Detecting Dementia through Retrospective Analysis of Routine Blog Posts by Bloggers with Dementia,
V. Masrani and G. Murray and T. Field and G. Carenini, ACL 2017 BioNLP Workshop, Vancouver, Canada.


### Preparing features

The repo contains slight variants of preprocessing scripts that you are already familiar with (from the search engine practical). They are listed below in alphabetical order:

* **mk_doc_vectors**: this one is a slight variant on *mk_category_vectors* in the search engine practical. It makes vectors for each blog post in the data, using TF-IDF features. Run with 

      python3 -W ignore mk_doc_vectors [words|ngrams]

(Depending on your chosen number of features, it may take a few minutes, especially if you have many features. So make yourself a cup of coffee...)

* **ngrams.py:** for each class, outputs a count file of all ngrams with a specific size. Run with 

      python3 ngrams.py [ngram size]

* **output_top_tfidfs.py**: for each class, outputs a TF-IDF file with the top *k* features for that class. Run with 

      python3 output_top_tfidfs.py [words|ngrams] [num_features_per_class].

* **words.py**: the equivalent of the ngrams.py script above, but using word tokens. Run with

      python3 words.py


Prepare your features using the above scripts in the right order. You will have to choose right at the beginning if you run on words or ngrams and use the appropriate scripts and flags for this (if running ngrams, use sizes in range 3-6). You should end up with a set of document vectors for both classes.


### Running the SVM

The SVM implementation used here comes from the [scikit-learn toolkit](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). It lets you build a classifier for the data using a range of different kernels and various parameter values.

Run the classifier with e.g.

    python3 classification.py --C=100 --kernel=linear

Instead of *linear*, you could also use *poly* for a polynomial kernel or *rbf* for an RBF Gaussian kernel. 100 is the C hyperparameter, which you can also change. When using *poly*, you can indicate the degree of the polynomial with e.g.

    python3 classification.py --C=100 --kernel=poly --degree=5

You will need to use a suitable split for training and testing your system. The program will ask you to choose how many training instances you want to use for each class. E.g.:

    class1 has 787 documents. How many do you want for training? 400
    class2 has 415 documents. How many do you want for training? 100

We then get the output of the SVM, the score over the test data. 

You can run the classifier with different kernels over your preprocessed data. Note the difference in number of support vectors selected by the classifier (that is the *nSV* value in the output). What do you conclude?



### Understanding your feature set

What are your results like? Too bad? Too good? Let's try to understand the behaviour of the system...

First, what happens when you change the number of features in *output\_top\_tf\_idfs.py*? Pay attention to your *recall* (the number of 'retained documents' at the end of running *mk_doc_vectors*, which shows the proportion of the dataset for which a vector could be built using the features at our disposal.

Second, go and look at your feature set in *data/vocab_file.txt*. Is everything as it should be? (For this part of the exercise, it may be easier to run your pipeline on words than ngrams.) 

Try to manually change the content of *data/vocab_file.txt* to what you think might be an appropriate feature set. What changes?


### Open-ended project

Have a look at the original paper for the data used in this practical and check which type of features were actually used in the experiments. Try to implement yourself some of their solutions.



