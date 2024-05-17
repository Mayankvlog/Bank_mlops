Banking MLOps Undertaking - Readme

This Streamlit web application is intended for the Banking MLOps Undertaking, pointed toward anticipating whether a client will buy into a term store in light of different information boundaries. Clients can include elements like age, work, conjugal status, schooling, balance, and so forth, through an instinctive sidebar interface. The application offers a determination of AI models including Irregular Backwoods, Calculated Relapse, SVM, and KNN for expectation. Perceptions, for example, histograms, bar plots, and dissipate plots give bits of knowledge into the dataset circulation and examples.

The undertaking uses MLflow for try following and reproducibility, logging boundaries, measurements, and antiques. Unit tests are carried out to guarantee the usefulness of client input elements and model preparation. Also, the application is containerized involving Docker for simple sending, and CI/Compact disc work processes are characterized involving CircleCI for computerized testing and organization.

# Run cammands

MLFLOW_TRACKING_URI=https://dagshub.com/Mayankvlog/Bank_mlops.mlflow \
MLFLOW_TRACKING_USERNAME=Mayankvlog \
MLFLOW_TRACKING_PASSWORD=a163dc11889cf8831384598f795e7c53cd54b1d2 \
streamlit run app.py

## Run cammands

docker run -d -p 8501:80 mayank035/toyotaapp:latest
docker ps -a
docker logs $Cointainer_id