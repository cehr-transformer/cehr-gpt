# this file was created by following
# https://github.com/pypa/sampleproject/blob/master/setup.py
# as example

from setuptools import setup, find_packages

setup(
    name='concept-embeddings',
    # This allows to use git/hg to auto-generate new versions
    use_scm_version={"root": ".", "relative_to": __file__},
    setup_requires=['setuptools_scm'],
    python_requires='>=3.6.0',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='development',
    packages=find_packages(where='.'),
    install_requires=['Keras>=2.0.8', 'numpy', 'tqdm', 'tensorflow', 'pandas', 'scikit-learn',
                      'xgboost'],
    tests_require=['pytest'],
    include_package_data=True,
)
