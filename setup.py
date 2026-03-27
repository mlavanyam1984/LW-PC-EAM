from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="lw-pc-eam",
    version="1.0.0",
    author="LW-PC-EAM Authors",
    description=(
        "Lightweight PatchCore with Explainable Attention Mechanism "
        "for Real-Time Industrial Anomaly Detection on Edge Devices"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/LW-PC-EAM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "anomaly detection", "industrial inspection", "PatchCore",
        "explainable AI", "edge deployment", "MobileNetV2",
    ],
    entry_points={
        "console_scripts": [
            "lwpceam-train=scripts.train_eval:main",
            "lwpceam-demo=scripts.inference_demo:main",
            "lwpceam-ablation=scripts.ablation_study:main",
        ],
    },
)
