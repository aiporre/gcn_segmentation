import os
import ftplib
import os

def upload_training(prefix,EPOCHS,lr,dataset_name, client="ftpclient",password="dextro",fig_dir='./fig',PORT = 2121, HOST = "213.206.185.111"):
    def upload(ftp, file):
        ext = os.path.splitext(file)[1]
        if ext in (".txt", ".htm", ".html"):
            ftp.storlines("STOR " + file, open(file))
        else:
            ftp.storbinary("STOR " + file, open(file, "rb"), 1024)
    
    ftp = ftplib.FTP()
    ftp.connect(HOST,PORT)
    ftp.login(client, password)
    os.chdir(fig_dir)
    upload(ftp, "{}_e{}_lr{}_ds{}_loss_history.png".format(prefix,EPOCHS,lr,dataset_name))
    upload(ftp, "{}_e{}_lr{}_ds{}_performance.png".format(prefix,EPOCHS,lr,dataset_name))
    os.chdir('../')
    upload(ftp, "{}-ds{}.pth".format(prefix,dataset_name))
    upload(ftp, "{}_e{}_lr{}_ds{}_lossall.npy".format(prefix,EPOCHS,lr,dataset_name))
    upload(ftp, "{}_e{}_lr{}_ds{}_measurements.npy".format(prefix,EPOCHS,lr,dataset_name))
    ftp.close()
    

def get_npy_files(path='./', extension='npy'):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(file))
    return files