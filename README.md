# The documentation of Ascend NPU for PyTorch(torch_npu)

This repository is a collection of documentation, api and stuff like that for torch_npu (Ascend NPU for PyTorch).

# Contributing

If you are interested in the project, please feel free to contribute it.

We use sphinx and rst file to create docs and you need to refer to rst syntax for help.
Here is how you can create a new docs:

1. Create a rst file. You can put it anywhere, but it must be better to classify it.
2. Add the rst into the index.rst

# Building locally

When the development is ready, you can check and test it locally by following the directives below:

1. Install the required dependencies:  

```bash
pip install -r requirements.txt
```

2. Building:  

```bash
make html
```

3. Server on localhost:  

```bash
python -m http.server -d _build/html 4000
```

Then open [localhost:4000](http://localhost:4000) in your browser.
