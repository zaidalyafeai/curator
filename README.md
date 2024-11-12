<p align="center">
  <a href="https://bespokelabs.ai/" target="_blank">
    <img src="./docs/Bespoke-Labs-Logo-on-Mint.png" width="50%" alt="Bespoke Labs Logo" />
  </a>
  <br /> <br />
</p>

<h1 align="center">Curator</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Data Curation for Post-Training for Structured Data Extraction</h3>
<p align="center">
  <a href="https://bespokelabs.ai/">
    <img alt="Site" src="https://img.shields.io/badge/Site-bespokelabs.ai-blue?link=https%3A%2F%2Fbespokelabs.ai"/>
  </a>
  <a href="https://twitter.com/bespokelabsai">
    <img src="https://img.shields.io/twitter/follow/bespokelabsai" alt="Follow on X" />
  </a>
</p>

# Bespoke Curator

Bespoke Labs Synthetic Data Curation Library

### Installation

```bash
pip install bespokelabs-curator
```

### Usage

```python
from bespokelabs import curator
import os

os.environ['OPENAI_API_KEY'] = 'sk-...' # Set your OpenAI API key here

poet = curator.Prompter(
    prompt_func=lambda: {
        "user_prompt": "Write a poem about the beauty of computer science"
    },
    model_name="gpt-4o-mini",
)

poem = poet()
print(poem.to_list()[0])
```

You can see more examples in the [examples](examples) directory.

To run the examples, make sure to set your OpenAI API key in the environment variable `OPENAI_API_KEY` by running `export OPENAI_API_KEY=sk-...` in your terminal.

## Bespoke Curator Viewer

To run the bespoke dataset viewer:

```bash
curator-viewer
```

This will pop up a browser window with the viewer running on `127.0.0.1:3000` by default if you haven't specified a different host and port.


Optional parameters to run the viewer on a different host and port:
```bash
>>> curator-viewer -h
usage: curator-viewer [-h] [--host HOST] [--port PORT] [--verbose]

Curator Viewer

options:
  -h, --help     show this help message and exit
  --host HOST    Host to run the server on (default: localhost)
  --port PORT    Port to run the server on (default: 3000)
  --verbose, -v  Enables debug logging for more verbose output
```

The only requirement for running `curator-viewer` is to install node. You can install them by following the instructions [here](https://nodejs.org/en/download/package-manager).

For example, to check if you have node installed, you can run:

```bash
node -v
```

If it's not installed, installing latest node on MacOS, you can run:

```bash
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
# download and install Node.js (you may need to restart the terminal)
nvm install 22
# verifies the right Node.js version is in the environment
node -v # should print `v22.11.0`
# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
```
