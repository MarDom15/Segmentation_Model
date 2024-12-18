
# Utilisez une image Python officielle en tant que base
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers requis dans le conteneur
COPY requirements.txt .
COPY new.py .
COPY smooth_tiled_predictions.py .
COPY dom.jpg .
COPY Resnet18_e25.hdf5 .
COPY Resnet18_e100.hdf5 .
COPY Resnet34_e25.hdf5 .
COPY Resnet34_e100.hdf5 .
COPY Resnet34_e300.hdf5 .
COPY Resnet50_e25.hdf5 .
COPY Resnet50_e100.hdf5 .
COPY Resnet50_e300.hdf5 .
COPY Resnet101_e25.hdf5 .
COPY Resnet101_e100.hdf5 .
COPY Resnet101_e300.hdf5 .

# Installez les dépendances définies dans requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel Streamlit s'exécute par défaut
EXPOSE 8501

# Commande pour exécuter l'application Streamlit lorsque le conteneur démarre
CMD ["streamlit", "run", "new.py"]
