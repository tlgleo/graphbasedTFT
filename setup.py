from setuptools import setup

setup(name='graph_basedTFT',
      version='0.0.1',
        packages=['graphbasedTFT'],
        install_requires=[
            'matplotlib>= 3.1.1',
            'networkx>= 2.3',
            'numpy>= 1.17.2',
            'scipy>= 1.6.0',
            'scikit-learn>=0.21.3'
        ]
)