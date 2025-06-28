from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='videoparquet',
    version='0.1.1',
    description='Convert Parquet files to video and back, inspired by xarrayvideo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/videoparquet',
    license='MIT',
    packages=find_packages(exclude=["tests*", "xarrayvideo*"]),
    install_requires=[
        'pyarrow',
        'pandas',
        'numpy',
        'ffmpeg-python',
        'scikit-learn',
    ],
    python_requires='>=3.7',
    keywords='parquet video ffmpeg scientific-data compression',
    project_urls={
        'Documentation': 'https://github.com/yourusername/videoparquet',
        'Source': 'https://github.com/yourusername/videoparquet',
        'Tracker': 'https://github.com/yourusername/videoparquet/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Multimedia :: Video',
        'Development Status :: 3 - Alpha',
    ],
) 