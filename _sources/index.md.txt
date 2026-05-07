.. image:: _static/images/logo-light.svg
   :width: 400
   :align: center
   :alt: judo
   :class: only-light

.. image:: _static/images/logo-dark.svg
   :width: 400
   :align: center
   :alt: judo
   :class: only-dark


# 🥋 Judo 🥋

<!-- prettier-ignore-start -->

.. toctree::
   :caption: Documentation
   :maxdepth: 10
   :hidden:
   :titlesonly:

   self
   quickstart
   interface/index
   docs
   faq
   api/index


|python| |nbsp| |test| |nbsp| |docs| |nbsp| |coverage|

.. |python| image:: https://img.shields.io/badge/Python-3.10%7C3.11%7C3.12%7C3.13-blue?logo=python&logoColor=white
   :alt: Supported Python versions
   :target: https://github.com/bdaiinstitute/judo

.. |test| image:: https://github.com/bdaiinstitute/judo/actions/workflows/test.yml/badge.svg
   :alt: Test status
   :target: https://github.com/bdaiinstitute/judo/actions/workflows/test.yml

.. |docs| image:: https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml/badge.svg
   :alt: Docs status
   :target: https://github.com/bdaiinstitute/judo/actions/workflows/docs.yml

.. |coverage| image:: https://codecov.io/gh/bdaiinstitute/judo/graph/badge.svg?token=3GGYCZM2Y2
   :alt: Test coverage
   :target: https://codecov.io/gh/bdaiinstitute/judo

.. |nbsp| unicode:: 0xA0
   :trim:


<!-- prettier-ignore-end -->

<p align="center">
  <img src="/judo/_static/images/banner.gif" alt="banner" width="800">
</p>

`judo` is a `python` package inspired by `mujoco_mpc <https://github.com/google-deepmind/mujoco_mpc>`_ that makes sampling-based MPC easy. Features include:

- 👩‍💻 A simple interface for defining custom tasks and controllers.
- 🤖 Automatic parsing of configs into a browser-based GUI, allowing real-time parameter tuning.
- 📬 Asynchronous interprocess communication using `dora <https://dora-rs.ai/>`_ for easy integration with your hardware.
- 🗂️ Configuration management with `hydra <https://hydra.cc/docs/intro/>`_ for maximum flexibility.

> ⚠️ **Disclaimer** ⚠️
>
> This code is released as a **research prototype** and is *not* production-quality software. It may contain missing features and potential bugs. The RAI Institute does **not** offer maintenance or support for this software. While we *may* accept pull requests for new features or bugfixes, we **cannot guarantee** timely responses to issues.
>
> The current release is also in **alpha**. We reserve the right to make breaking changes to the API and configuration system in future releases. We will try to minimize these changes, but please be aware that they may occur.
