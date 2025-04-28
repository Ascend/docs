# The Doc of Ascend Supported Open Source project

This repository is a collection of documentation for Ascend supported open source projects.

# Structure

- `index.rst`: the index of the opensource project doc.
- `quick_start.rst`: **5 minutes** or **600 words**, quickstarts enable people to quickly complete a discrete, focused task by illustrating a workflow with **only essential steps**. See also in [Github quickstart guide](https://docs.github.com/en/contributing/style-guide-and-content-model/quickstart-content-type).
- `install.rst`: installation guide, including common installation (such as binary, pypi, conda) for developers and a quick installation verification
- `tutorial.rst`: more complex than quickstart, help people learn about **products** and solve **real world problems** by guiding them through **the entire workflow to complete a task**. See also in [Github tutorial guide](https://docs.github.com/en/contributing/style-guide-and-content-model/tutorial-content-type).
- `faq.rst`: A frequently asked questions (FAQ) list.

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
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu
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
