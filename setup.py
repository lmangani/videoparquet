from setuptools import setup, find_packages

setup(
    name='videoparquet',
    version='0.1.0',
    description='Convert Parquet files to video and back, inspired by xarrayvideo',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'pyarrow',
        'pandas',
        'numpy',
        'ffmpeg-python',
    ],
    python_requires='>=3.7',
) 