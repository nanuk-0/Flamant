import os

#PROJECT_HOME = f"{os.environ['cours']}mémoire/03-code/memoire/"
PROJECT_HOME = r"C:\Users\louis\PycharmProjects\Master_Thesis\Flamant"

PORT = 5000
IP = uri="127.0.0.1"
def MLFLOW_URI(port=PORT, ip=IP):
    return f"http://{ip}:{port}"