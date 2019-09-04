import os
import ftplib
import os

def upload_training(prefix,EPOCHS,lr,dataset_name, client="ftpclient",password="dextro",fig_dir='./fig',PORT = 2121, HOST = "213.206.185.111"):
    def upload(ftp, file):
        if not os.path.exists(file):
            print("File '", file, "' doesn't exist")
            return
        ext = os.path.splitext(file)[1]
        try:
            if ext in (".txt", ".htm", ".html"):
                with open(file) as f:
                    ftp.storlines("STOR " + file, f)
            else:
                with open(file, "rb") as f:
                    ftp.storbinary("STOR " + file, f, 1024)
        except Exception as e:
            print('EXCEPTION FTP: ', e)

    ftp = ftplib.FTP()
    ftp.connect(HOST,PORT)
    ftp.login(client, password)
    os.chdir(fig_dir)
    print('uploading images')

    upload(ftp, "{}_e{}_lr{}_ds{}_loss_history.png".format(prefix,EPOCHS,lr,dataset_name))
    upload(ftp, "{}_e{}_lr{}_ds{}_performance.png".format(prefix,EPOCHS,lr,dataset_name))
    os.chdir('../')
    print('uploading model')
    upload(ftp, "{}-ds{}.pth".format(prefix,dataset_name))
    print('uploading checkpoint')
    upload(ftp, "{}_e{}_lr{}_ds{}_checkpoint.npy".format(prefix,EPOCHS,lr,dataset_name))
    print('uploading metrics')
    upload(ftp, "{}_e{}_lr{}_ds{}_lossall.npy".format(prefix,EPOCHS,lr,dataset_name))
    upload(ftp, "{}_e{}_lr{}_ds{}_measurements.npy".format(prefix,EPOCHS,lr,dataset_name))
    ftp.close()
    print('ftp client closed. Done uploading data.')
    

def get_npy_files(path='./', extension='npy'):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(file))
    return files