# ChainerMN seq2seq example

A sample implementation of seq2seq model.

## Data download and setup

First, go to http://www.statmt.org/wmt15/translation-task.html#download and donwload necessary dataset.
Let's assume you are in a working directory called `$WMT_DIR`.

```bash
$ WMT_DIR=$HOME/path/to/your/data
$ cd $WMT_DIR
$ wget http://www.statmt.org/wmt10/training-giga-fren.tar
$ wget http://www.statmt.org/wmt15/dev-v2.tgz
$ tar -xf training-giga-fren.tar
$ tar -xf dev-v2.tgz
$ ls 
dev/  dev-v2.tgz  giga-fren.release2.fixed.en.gz  giga-fren.release2.fixed.fr.gz  training-giga-fren.tar

```

Next, you need to install required packages.

```bash
$ pip install nltk progressbar2
```

## Run

```bash
$ cd $CHAINERMN
$ mpiexec -n 2 python examples/seq2seq/seq2seq.py --gpu -c $WMT_DIR -i $WMT_DIR --out result.tmp
```
