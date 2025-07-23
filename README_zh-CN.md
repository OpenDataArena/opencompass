<div align="center">
  <img src="docs/zh_cn/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![][github-release-shield]][github-release-link]
[![][github-releasedate-shield]][github-releasedate-link]
[![][github-contributors-shield]][github-contributors-link]<br>
[![][github-forks-shield]][github-forks-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]
[![][github-license-shield]][github-license-link]

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐官方网站](https://opencompass.org.cn/) |
[📖数据集社区](https://hub.opencompass.org.cn/home) |
[📊性能榜单](https://rank.opencompass.org.cn/home) |
[📘文档教程](https://opencompass.readthedocs.io/zh_CN/latest/index.html) |
[🛠️安装](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔报告问题](https://github.com/open-compass/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

[![][github-trending-shield]][github-trending-url]

</div>


> \[!IMPORTANT\]
>
> **收藏项目**，你将能第一时间获取 Data LeaderBoard 的最新动态～⭐️

<details>
  <summary><kbd>Star History</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&type=Date">
  </picture>
</details>

## 🧭	欢迎



<p align="right"><a href="#top">🔝返回顶部</a></p>



### 📂 数据准备

#### 提前离线下载

OpenCompass支持使用本地数据集进行评测，数据集的下载和解压可以通过以下命令完成：

```bash
# 下载数据集到 data/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

#### 从 OpenCompass 自动下载

我们已经支持从OpenCompass存储服务器自动下载数据集。您可以通过额外的 `--dry-run` 参数来运行评估以下载这些数据集。
目前支持的数据集列表在[这里](https://github.com/open-compass/opencompass/blob/main/opencompass/utils/datasets_info.py#L259)。更多数据集将会很快上传。

#### (可选) 使用 ModelScope 自动下载

另外，您还可以使用[ModelScope](www.modelscope.cn)来加载数据集：
环境准备：

```bash
pip install modelscope
export DATASET_SOURCE=ModelScope
```



<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🏗️ ️评测

在确保按照上述步骤正确安装了 当前版本OpenCompass 并准备好了数据集之后，现在您可以开始使用 OpenCompass 进行首次Data Deaderboard评估！

## 🔜 路线图





## 🤝 致谢



## 🖊️ 引用




<p align="right"><a href="#top">🔝返回顶部</a></p>

[github-contributors-link]: https://github.com/open-compass/opencompass/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/opencompass?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/opencompass/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/opencompass?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/opencompass/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/opencompass?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/opencompass/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/opencompass?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/open-compass/opencompass/releases
[github-release-shield]: https://img.shields.io/github/v/release/open-compass/opencompass?color=369eff&labelColor=black&logo=github&style=flat-square
[github-releasedate-link]: https://github.com/open-compass/opencompass/releases
[github-releasedate-shield]: https://img.shields.io/github/release-date/open-compass/opencompass?labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/opencompass/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/opencompass?color=ffcb47&labelColor=black&style=flat-square
[github-trending-shield]: https://trendshift.io/api/badge/repositories/6630
[github-trending-url]: https://trendshift.io/repositories/6630
