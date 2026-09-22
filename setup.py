import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='openimis-be-biometric_verification',
    version='0.1.4',
    packages=find_packages(),
    include_package_data=True,
    license='LGPL-3.0',
    description='Biometric identity verification module for openIMIS.',
    long_description=README,
    url='https://openimis.org/',
    install_requires=[
        'django',
        'djangorestframework',
        'graphene-django',
        'openimis-be-core',
        'cryptography',
        'numpy',
    ],
    extras_require={
        # Local inference — pip install "openimis-be-biometric_verification[deepface]"
        # Note: opencv-python-headless MUST be installed before deepface
        # to avoid deepface installing opencv-python (with GUI deps)
        'deepface': [
            'opencv-python-headless>=4.9.0',  # Install FIRST to satisfy opencv requirements
            'numpy>=2.0',
            'tf-keras>=2.17.0',
            'deepface>=0.0.93',
        ],
        # AWS Rekognition — pip install "openimis-be-biometric_verification[aws]"
        'aws': [
            'boto3',
        ],
        # Azure Face API — pip install "openimis-be-biometric_verification[azure]"
        'azure': [
            'azure-cognitiveservices-vision-face',
            'msrest',
        ],
        # pgvector ANN index — pip install "openimis-be-biometric_verification[pgvector]"
        # Used by the optional biometric_pgvector app, not by biometric itself.
        'pgvector': [
            'pgvector>=0.3',
        ],
    },
    classifiers=[
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
    ],
)
