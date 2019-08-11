from setuptools import setup

setup(
        name='pytorch_autograd_checkpointing',
        version='0.1.0',
        description='stuff',
        url='https://github.com/shirazb/pytorch-autograd-checkpointing',
        author='Shiraz Butt',
        author_email='shiraz.b@icloud.com',
        license='MIT',

        package_dir={'pytorch_autograd_checkpointing': 'src'},
        packages=['pytorch_autograd_checkpointing'],
        
        install_requires=['torch>=1.1.0'],
        #setup_requires=['pytest-runner'],
        #tests_require=['pytest'],

        zip_safe=False # ??
)
