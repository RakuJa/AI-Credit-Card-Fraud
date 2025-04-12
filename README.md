# AI-Credit-Card-Fraud

## Training

### Where to find the dataset

The dataset can be found at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and should be downloaded and put inside the /python/dataset/ folder naming the file "data.arff"

### How to train
Setup environmental variables to have granular control over the training process.
For the first run it's recommended to keep everything enabled

After that run the main.py file.

## GUI and visualization
### Required dependencies

sudo pacman -Syu libclang-dev libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev

### Run it
```bash
cargo run --release
```