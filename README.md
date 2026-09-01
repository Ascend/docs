# The Doc of Ascend Supported Open Source project

This repository is a collection of documentation for Ascend supported open source projects.

# Structure

## New Doc

- `index.rst`: the index of the opensource project doc.
- `quick_start.rst`: **5 minutes** or **600 words**, quickstarts enable people to quickly complete a discrete, focused task by illustrating a workflow with **only essential steps**. See also in [Github quickstart guide](https://docs.github.com/en/contributing/style-guide-and-content-model/quickstart-content-type).
- `install.rst`: installation guide, including common installation (such as binary, pypi, conda) for developers and a quick installation verification
- `tutorial.rst`: more complex than quickstart, help people learn about **products** and solve **real world problems** by guiding them through **the entire workflow to complete a task**. See also in [Github tutorial guide](https://docs.github.com/en/contributing/style-guide-and-content-model/tutorial-content-type).
- `faq.rst`: A frequently asked questions (FAQ) list.

## Documentation from Official Project Sites

Some communities already maintain Ascend/NPU documentation on their own sites. Those projects are linked from the homepage cards in `index.rst` instead of being built locally. Examples: verl, LLaMA-Factory, ms-swift, VeOmni, vllm-ascend, Triton-Ascend, DeepSpeed, ONNX Runtime, SGLang, LMDeploy, ROLL, and Twinkle.

To add such a project, create a lightweight `sources/<project>/index.rst` with links to the official docs, and add it to the hidden toctree in `index.rst` so it appears in the sidebar.

## Local Build Prerequisites

Before `make html`, the build fetches `ascend_config.json` via `scripts/fetch_ascend_data.py` (see `fetch-config` in `./Makefile`). No git submodules or document copying is involved.

# Contributing

If you are interested in the project, please feel free to contribute it.

We use sphinx and rst file to create docs and you need to refer to rst syntax for help.
Here is how you can create a new docs:

1. Create a rst file. You can put it anywhere, but it must be better to classify it.
2. Add the rst into the index.rst

## PR preview

Every pull request gets a rendered preview of the whole site. Once the `Github Pages`
build finishes (about 10 minutes), a bot comments the link on the PR:

```
https://ascend.github.io/docs/pr-preview/pr-<PR number>/
```

It is rebuilt on every push and deleted when the PR is closed. If the build fails,
no preview is published — check the failure in the Actions tab.

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
