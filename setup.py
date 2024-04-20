from setuptools import setup, find_packages

setup(
    name='Python-DRL',
    version='1.0.1',
    packages=find_packages(),
    description='A pytorch-based deep reinforcement learning package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='lyuanzhi',
    author_email='lyuanzhi862066195@outlook.com',
    url='https://github.com/lyuanzhi/Python-DRL',
    install_requires=[
        'numpy~=1.23.0',
        'torch>=2.0.1',
        'gym~=0.26.2',
        'opencv-python>=4.5.5.62',
        'pygame~=2.5.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

