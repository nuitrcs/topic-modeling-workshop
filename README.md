# topic-modeling-workshop
## Materials for Topic Modeling Workshop at Northwestern University.

The following files are included in the workshop materials.

<pre>
        bbc-topics.ipynb                Topic modeling of BBC news articles using
                                        Latent Dirichlet Allocation.
        abc-news-headlines.ipynb        Topic modeling of Australian Broadcast News
                                        headlines using Latent Dirichlet Allocation
                                        and Non-negative Matrix Factorization.

        NMF-parameter-selection.ipynb   \
        NMF-text-preprocessing.ipynb     | Topic modeling of BBC news articles.
        NMF-topic-modeling.ipynb        /

        climate-tweets.ipynb            Topic modeling of climate change related tweets.

        README.md                       What you're reading now.

        references.html                 \
        references.md                    | General references and pointers to sources for data
                                         | and code used in this workshop.
                                        /

        requirements.txt                Python requirements for running workshop code.

        topic-model-slides.pptx         Presentation slides for workshop.
        topic-modeling-workshop.yaml    Conda requirements for generating online environment
                                        for users on workshop server.
</pre>

## Running the Jupyter notebooks during the workshop

You will be running the Python notebooks on a server,
so you do not need to install any software.   Your username
and password will be provided at the beginning of the workshop.

## Running the notebooks on your own machine

To run the Python notebooks on your own machine, clone the
topic modeling repository using git.  You will need to install
git before you can do this.

On most systems the git command to clone the repository looks like:

        git clone https://github.com/nuitrcs/topic-modeling-workshop.git

You may also use your favorite graphical git client.

Next, install Python if you have not already done so.   We highly
recommend you install Anaconda Python for your system.  The workshop
notebooks are written in Python3, so please make sure you install
that version of Anaconda Python.

Once you have installed Anaconda move into the directory
to which you cloned the topic modeling workshop files and
type the following at a command line:

        pip install -r requirements.txt

(You may need to run

        pip3 install -r requirements.txt

instead in some cases.)

This installs the Python modules needed to run the
workshop notebooks.

You may want to install the workshop modules into a separate
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
If you are using Anaconda, you can instead
[create a virtual environment using conda.](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

To install the extra data needed by the nltk module,
type the following at a command line:

        python -m nltk.downloader popular

To execute a notebook, move to the directory into
which you cloned the workshop files, and type
the following at a command line:

        jupyter notebook

This brings up the jupyter web application from which
you can run and edit the workshop notebooks.  If you created
a virtual environment for the workshop materials, make to
activate the environment before invoking jupyter,


