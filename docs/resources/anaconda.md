<div align="center">

Anaconda
=============================

<div align="center">
  <a href="../../data/pdf/conda_cheatsheet.pdf">Conda Cheatsheet</a> •
  <a href="../../data/pdf/pip_cheatsheet.pdf">PIP Cheatsheet</a>
</div>
</div>

## Installation

<details open>
<summary><b style="font-size:18px">Ubuntu and MacOS</b></summary>

If the path is not automatically add to the `.bashrc` file. So in this case, 
we can manually add it as follows:

```shell
sudo nano ~/.basrc        # Ubuntu
sudo nano ~/.bash_profile # MacOS
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

</details>

<details open>
<summary><b style="font-size:18px">Windows</b></summary>

Make sure to add the path to `anaconda3/bin` and `anaconda3/lib` in `PATH` environment.
</details>
