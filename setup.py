from setuptools import setup, find_packages

setup(
        name="daphnia",
        version="0.1",
        packages=find_packages(),
        install_requires=['Click',
            'numpy==1.13.0',
            'scipy==0.19.1',
            'pandas==0.20.2',
            'opencv-python==3.4.0.12',
            'scikit-image==0.13.0',
            'openpyxl==2.5.3'
            ],
        entry_points='''
            [console_scripts]
            daphnia=daphnia.scripts.run_daphnia:daphnia
        ''',
)
