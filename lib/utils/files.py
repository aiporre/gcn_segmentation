import os
import ftplib
import os

def upload_training(prefix_model, prefix,EPOCHS,lr, client="ftpclient",password="dextro",fig_dir='./fig',PORT = 2121, HOST = "213.206.185.111",h5format=False):
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

    upload(ftp, "{}_loss_history.png".format(prefix))
    upload(ftp, "{}_performance.png".format(prefix))
    os.chdir('../')
    print('uploading model')
    if h5format:
        upload(ftp,'./{}.pth'.format(prefix_model))
    else:
        upload(ftp, "{}.pth".format(prefix_model))
    print('uploading checkpoint')
    upload(ftp, "{}_checkpoint.npy".format(prefix))
    print('uploading metrics')
    upload(ftp, "{}_lossall.npy".format(prefix))
    upload(ftp, "{}_measurements.npy".format(prefix))
    ftp.close()
    print('ftp client closed. Done uploading data.')
    

def get_npy_files(path='./', extension='npy'):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(file))
    return files