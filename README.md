# Amesa

## Table of Content

- [Developer getting started guide](#developer-getting-started-guide)
  - [A. Set Your SSH Key](#a-set-your-ssh-key)
  - [B. Clone the Repository](#b-clone-the-repository)
  - [C. Install PyEnv and Set Up Your Environment](#c-install-pyenv-and-set-up-your-environment)

## Developer getting started guide

To get started locally in development, follow these steps:

### A. Set your SSH key

If you already have an SSH key, skip to the [third step and install PyEnv](#c-install-pyenv-and-set-up-your-environment) if it's not already installed. Otherwise, follow these steps to set up your SSH key.

#### 1. Generate a new SSH key pair and press enter

```sh
ssh-keygen
```

#### 2. Print the public key

```sh
cat ~/.ssh/{your_public_key}.pub
```

#### 3. Save your SSH key

Go to GitHub settings and add your ssh key.

### B. Clone the repository.

Clone the repository using SSH.

```sh
git clone git@github.com:Composabl/squad-build.git
```

### C. Install PyEnv and Set Up your environment

#### 1.1 Linux/Unix Automatic Installer

The Homebrew option from the [MacOS section](#macos) below also works if you have Homebrew installed.

```sh
curl -fsSL https://pyenv.run | bash
```

#### 1.2 MacOS

The options from the [Linux section](#11-linuxunix-automatic-installer) above also work, but Homebrew is recommended for basic usage.

##### Install PyEnv using [Homebrew](https://brew.sh)

```sh
brew update
brew install pyenv
```

#### 2. Set up your shell environment for Pyenv

The below setup should work for the vast majority of users for common use cases.

##### Bash

Stock Bash startup files vary widely between distributions in which of them source
which, under what circumstances, in what order and what additional configuration they perform.
As such, the most reliable way to get Pyenv in all environments is to append Pyenv
configuration commands to both `.bashrc` (for interactive shells)
and the profile file that Bash would use (for login shells).

1. First, add the commands to `~/.bashrc` by running the following in your terminal:

   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
   ```

2. Then, if you have `~/.profile`, `~/.bash_profile` or `~/.bash_login`, add the commands there as well.
   If you have none of these, create a `~/.profile` and add the commands there.
   - to add to `~/.profile`:
     ```bash
     echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
     echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
     echo 'eval "$(pyenv init - bash)"' >> ~/.profile
     ```
   - to add to `~/.bash_profile`:
     ```bash
     echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
     echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
     echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile
     ```

**Bash warning**: There are some systems where the `BASH_ENV` variable is configured
to point to `.bashrc`. On such systems, you should almost certainly put the
`eval "$(pyenv init - bash)"` line into `.bash_profile`, and **not** into `.bashrc`. Otherwise, you
may observe strange behaviour, such as `pyenv` getting into an infinite loop.
See [#264](https://github.com/pyenv/pyenv/issues/264) for details.

##### Zsh

```bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
  echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
  echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc
```

If you wish to get Pyenv in noninteractive login shells as well, also add the commands to `~/.zprofile` or `~/.zlogin`.

#### 3. Restart your shell

for the `PATH` changes to take effect.

```sh
exec "$SHELL"
```

#### 4. Install a python version

To install additional Python versions, use `pyenv install`

For example, to download and install Python 3.10.4, run:

```sh
pyenv install 3.10.4
```

Running `pyenv install -l` gives the list of all available versions.

#### 5. Create your virtual environment and activate it

Set the python version you are going to use for the current directory (this creates a .python-version file).

```sh
pyenv local 3.10.4
```

Then create the python environment and activate it.

```sh
python -m venv .venv
source .venv/bin/activate
```
