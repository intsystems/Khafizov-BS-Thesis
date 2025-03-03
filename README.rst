|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Адаптивное сжатие в распределенной оптимизации
    :Тип научной работы: НИР
    :Автор: Фанис Адикович Хафизов
    :Научный руководитель: к.ф.-м.н., Безносиков Александр Николаевич

Abstract
========

В данной работе рассматривается проблема распределённого обучения больших моделей (например, современных нейросетей), когда вычисления необходимо распараллеливать между несколькими устройствами. Основная сложность в таких системах заключается в высокой стоимости коммуникации при передаче больших объёмов градиентов. Мы предлагаем семейство операторов адаптивного сжатия, которые учитывают важность координат и тем самым снижают трафик, сохраняя качество сходимости. В экспериментальной части показано, что предлагаемые операторы могут работать не хуже классических вариантов RandK и TopK, а в ряде случаев достигают сопоставимого качества с TopK.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
