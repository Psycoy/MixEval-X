from setuptools import setup, find_packages

setup(
    name='mixeval_x',
    version='1.0.0',
    author='MixEval-X team',
    author_email='jinjieni@nus.edu.sg',
    packages=find_packages(),
    install_requires=[
        'transformers==4.42.3',
        'tiktoken==0.6.0',
        'fschat==0.2.36',
        'SentencePiece==0.2.0',
        'accelerate==0.30.1',
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'hf_transfer==0.1.6',
        'openai==1.30.5',
        'httpx==0.27.0',
        'nltk==3.8.1',
        'numpy==1.26.3',
        'tqdm==4.66.4',
        'protobuf==4.25.3',
        'python-dotenv==1.0.1',
        'anthropic==0.28.0',
        'google-generativeai==0.5.4',
        'google-cloud-aiplatform==1.53.0',
        'dashscope==1.19.2',
        "prettytable"
    ],
    package_data={
    },
    entry_points={
    },
    url='https://mixeval-x.github.io/',
    license='License :: OSI Approved :: MIT License',
    description='A real-world any-to-any benchmark and eval suite for models with diverse input and output modalities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
