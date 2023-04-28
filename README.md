# Predictive Keyboard for Multi-Dialect Languages

While there are many computational resources for many of the major languages, it remains that the majority of languages today do not have such support. This, combined with the fact that many of the 7,000+ languages on this planet do not have ample resources to train large language models, are why we decided to create this repo.

**How to Train**

To train this model, one simply copies the files into a local directory. Once one is there, they can replace the files in the `traindata` folder with those of their choosing. Each of the current files are for a single Sardinian dialect (sc-comu, sc-camp, sc-logu, and sc-nugo) and are named accordingly. The model uses the file names (excluding the `.txt`) as the predicted variant names.

Once ready to begin, navigate to the directory that contains `train.py` and run it using `$python3 train.py`. This will take some time but eventually it will produce a `.net` file which is the model.

**How to Test**

To test the model is very simple; if you are using new data, replace the testing dataset in the folder labeled `testdata`, otherwise test the model using `$python3 test.py model.net`. Change the name of the `.net` file if you had changed it.