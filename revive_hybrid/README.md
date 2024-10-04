# Installation

### Requirements

-   Linux x86\_64
-   [Python](https://python.org/): v3.7.0+ / v3.8.0+ / v3.9.0+
-   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (If NVIDIA
    CUDA GPU device is available.)

###  Install REVIVE

You can install the latest sversion of the from a cloned Git repository:

    $ git clone https://agit.ai/Polixir/revive
    $ cd revive
    $ pip install -e .

You can also view and download all versions of the code package through the releases page.

**Releases Page :** https://agit.ai/Polixir/revive/releases

Or pull the lastest image contains SDK and its runtime environment from Docker Hub :

    $ docker pull polixir/revive-sdk

###  Register an Account

The REVIVE SDK library is developed by Polixir. We encrypt and protect a few algorithm modules, which have their own intellectual property rights, and you can register an account to use the features of the full algorithm package.

The process  can be divided into the following two steps:

**Step 1**. Visit the REVIVE Website to register an account.

**REVIVE Website :** <https://www.revive.cn>

**Step 2**. Configure registered account information.

1.  Open the `config.yaml` file in your home directory (for example,`/home/your-user-name/.revive/.config.yaml`) in a text editor. After installing REVIVE SDK, the config.yaml file will be generated automatically.
2.  Get the accesskey from the [user center of the REVIVE Website](https://revive.cn/user-center/service) and fill it into the `config.yaml`.

``` {.yaml}
accesskey: xxxxxxxxx
```

