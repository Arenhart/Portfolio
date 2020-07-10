from setuptools import setup, find_packages

setup(
  name = 'porous_media_analyzer',
  packages=find_packages(),
  version = '0.1.1',
  license='MIT',  
  description = 'Implements a selection of porous media analysis methods, includes a simple GUI',
  author = 'Rafael Arenhart',
  author_email = 'rafael.arenhart@gmail.com',
  url = 'https://github.com/Arenhart/Portfolio/tree/master/porous_media_analyzer',
  install_requires=['Pillow', 'matplotlib', 'numpy', 'numpy_stl', 'scikit_image', 'scipy'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)