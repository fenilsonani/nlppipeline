# NLP Pipeline Installation Guide

## Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd nlppipeline
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download language models**
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

5. **Run the demo**
```bash
python demo.py
```

## Verification

Run the test scripts to verify installation:

```bash
# Quick functionality test
python test_quick.py

# Feature showcase
python showcase.py
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings (optional)

## Performance Notes

- First run will download BERT models (~400MB)
- Initial model loading takes 2-3 seconds
- Subsequent predictions are much faster (~50-200ms)

## Troubleshooting

### SSL Warning
If you see SSL warnings, they can be safely ignored for local development.

### Memory Issues
Reduce batch size in configuration if running on limited memory:
```python
config.model.batch_size = 16  # Default is 32
```

### Missing Dependencies
If Kafka/Spark features are needed:
```bash
pip install confluent-kafka pyspark
```

## Docker Deployment

```bash
# Build image
docker build -t nlp-pipeline .

# Run with docker-compose
docker-compose up
```

## Next Steps

- See `README.md` for usage examples
- Check `docs/` for detailed documentation
- Run `pytest tests/` for full test suite