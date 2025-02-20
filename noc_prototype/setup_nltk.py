import nltk

def download_nltk_data():
    """Download required NLTK data packages"""
    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'universal_tagset'
    ]
    
    for package in required_packages:
        print(f"Downloading {package}...")
        nltk.download(package)

if __name__ == "__main__":
    download_nltk_data() 