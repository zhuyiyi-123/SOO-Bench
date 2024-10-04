from setuptools import setup, find_packages

setup(
    name='soo_bench',
    version='0.2.0', 
    # author='111', 
    # author_email='111.com', 
    description='Benchmarks for Evaluating the Stability of Offline Black-Box Optimization',  
    long_description=open('README.md', encoding='utf-8').read(), 
    long_description_content_type='text/markdown',  
    py_modules=['soo_bench'],
    packages=find_packages(), 
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8', 
    install_requires=[
        'revive==0.7.3'
    ],
    
)
