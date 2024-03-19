from setuptools import setup

install_requires = [
    'torch',
    'hydra-core',
    'lightning',
    'omegaconf',
    ]

# Get version from the module
with open('hypatorch/__init__.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split('=')[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    name='hypatorch',
    version=version,
    description='HypaTorch: A library for abstract and visual model configuration',
    author='Altavo GmbH',
    url='https://github.com/Altavo/hypatorch/',
    license='Apache 2.0',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['hypatorch'],
    install_requires=install_requires
)