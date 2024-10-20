from setuptools import setup, find_packages

setup(
    name='mixeval_x',
    version='1.0.0',
    author='MixEval-X team',
    author_email='jinjieni@nus.edu.sg',
    packages=find_packages(),
    install_requires=[
        'tiktoken==0.6.0',
        'pandas==2.2.2',
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
        'Pillow'
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
