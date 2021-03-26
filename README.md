# SVMs for Alzheimer detection

This is an implementation of an SVM classifier for Alzheimer detection, given a set of blog posts written by patients and family members. It will give us the opportunity to have a look at the output of an SVM, and also to understand how to be conifident about our results (spoiler... the main question will be: are our features sensible?)

### The data

The data is provided to you in a processed form, in the *data* directory. The *class1* folder contains data from Alzheimer patients, while the *class2* folder contains data from the control group. 

If you want to know where the data came from, note that the raw blog posts can be extracted using the script here: [https://github.com/vmasrani/blog_corpus](https://github.com/vmasrani/blog_corpus). This dataset was produced for the following publication:

Detecting Dementia through Retrospective Analysis of Routine Blog Posts by Bloggers with Dementia,
V. Masrani and G. Murray and T. Field and G. Carenini, ACL 2017 BioNLP Workshop, Vancouver, Canada.


### Preparing features

The repo contains slight variants of scripts that you are already familiar with (from the search engine practical):

* **ngrams.py:** for each class, outputs a count file of all ngrams with a specific size. Run with 

      python3 ngrams.py [ngram size]

* **words.py**: the equivalent of the ngrams.py script, but using word tokens. Run with

      python3 words.py

* **output_top_tfidfs.py**: for each class, outputs a tf\_idf file with the top k features for that class. Run with 

      python3 output_top_tfidfs.py [words|ngrams] [num_features_per_class].

* **mk_doc_vectors**: this one is a slight variant on *mk_category_vectors* in the search engine practical. It makes vectors for each blog post in the data, using the features from the tf\_idf files. Run with 

      python3 mk_doc_vectors [words|ngrams]

(it will take a few minutes, especially if you have many features. So make yourself a cup of coffee...)

Prepare your features using the above scripts (for ngrams, use sizes in range 3-6). You should end up with a set of document vectors for both classes.


### Running the SVM

The SVM implementation used here comes from the [scikit-learn toolkit](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). It lets you build a classifier for the data using a range of different kernels and various parameter values.

Run the classifier with e.g.

    python3 classification.py --C=100 --kernel=linear

Instead of *linear*, you could also use *poly* for a polynomial kernel or *rbf* for an RBF Gaussian kernel. 100 is the C hyperparameter, which you can also change. When using *poly*, you can indicate the degree of the polynomial with e.g.

    python3 classification.py --C=100 --kernel=poly --degree=5

The program will ask you to choose how many training instances you want to use for each class. E.g.:

    class1 has 787 documents. How many do you want for training? 400
    class2 has 415 documents. How many do you want for training? 100

We then get the output of the SVM, the score over the test data. Two confusion matrices will also be printed as .png in your directory, showing the errors made by the system (one version shows error frequencies, and the other percentages of errors).




### Inspecting the number of support vectors

Run the classifier with different kernels and notice the difference in number of support vectors selected by the classifier (that is the *nSV* value in the output). What do you conclude?


### Understanding the effect of class distribution

Try and play with the number of documents you use for training in each class. What do you notice? Is that what you expected?


### Understanding your feature set

What are your results like? Too bad? Too good? Go and look at your feature set in *data/vocab_file.txt*. Is everything as it should be? 

What happens when you change the number of features in *output\_top\_tf\_idfs.py*? Do you now see why you are getting the kind of results you're seeing? Pay attention to your *recall* (the number of 'retained documents' at the end of running *mk_doc_vectors*, which shows the proportion of the dataset for which a vector could be built using the features at our disposal.

Change your feature set to have more sensible results... You can split the work between yourselves and implement / test different possible ideas.



