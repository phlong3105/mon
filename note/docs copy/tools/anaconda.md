---
layout      : default
title       : Anaconda
parent		: Tools
has_children: false
has_toc     : false
permalink   : /tools/anaconda
---

# Anaconda

[Conda Cheatsheet](data/conda_cheatsheet.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }
[PIP Cheatsheet](data/pip_cheatsheet.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

## Installation

### Ubuntu and macOS

If the path is not automatically add to the `.bashrc` file. So in this case,
we can manually add it as follows:

```shell
sudo nano ~/.basrc         # Ubuntu
sudo nano ~/.bash_profile  # MacOS
```

Paste the following text at the end of the file:

```shell
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/home/longpham/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/home/longpham/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/longpham/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/home/longpham/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<
```

### Windows

Make sure to add the path to `anaconda3/bin` and `anaconda3/lib` in `PATH`
environment.
