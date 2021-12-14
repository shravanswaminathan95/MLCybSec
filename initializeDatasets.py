import os
import hashlib
import json
#import keras
#from keras.datasets import mnist
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from binascii import unhexlify, hexlify

dumpJsonFile = 'validity.json'


def generate_md5_hash (file_data):
    md5_hash = hashlib.md5(file_data)
    f_id = md5_hash.hexdigest()
    return str(f_id)


def encryptFilesAndStore(dir):
    dictDumpout = {}
    dir = os.sep.join([dir, "MNIST", "raw"])

    keyObject = inputKey()

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file != dumpJsonFile:
                openedFile = open(os.sep.join([root, file]), 'rb')
                openedFileContent = openedFile.read()
                dictDumpout[file] = keyObject.encrypt(unhexlify(generate_md5_hash(openedFileContent)))
                dictDumpout[file] = dictDumpout[file].decode()

    with open(os.sep.join([dir, dumpJsonFile]), 'w', encoding='utf-8') as dumpFile:
        json.dump(dictDumpout, dumpFile, ensure_ascii=False, indent=4)


def decryptFilesAndVerify(dir):
    dir = os.sep.join([dir, "MNIST", "raw"])
    retval = 0

    keyObject = inputKey()

    jFile = open(os.sep.join([dir, dumpJsonFile]), 'r')
    jFileContent = jFile.read()
    x = eval(jFileContent)
    dictDumpReadIn = json.dumps(x)
    dictDumpReadIn = json.loads(dictDumpReadIn)
    jFile.close()

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file != dumpJsonFile:
                openedFile = open(os.sep.join([root, file]), 'rb')
                openedFileContent = openedFile.read()
                hashContent = unhexlify(generate_md5_hash(openedFileContent))
                try:
                    decryptedHash = keyObject.decrypt(dictDumpReadIn[file].encode())

                    if decryptedHash == hashContent:
                        print("Valid file - ", file)
                    else:
                        print("Invalid file - ", file)
                except:
                    retval+=1
                    print("FAILURE - Cryptographical value verification failed")
                    print(".....Check if {} exists in the directory".format(file))
                    print(".....{} is not yet authenticated".format(file))

    return retval

def inputKey():
    # Here we use Symmetric encryption using AES. The Fernet class of the cryptography library implements AES
    print("Enter the key to encrypt the hashes:")
    key = input("\\\\>").encode()

    salt = b'SALT'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    _key = base64.urlsafe_b64encode(kdf.derive(key))

    f = Fernet(_key)

    return f


if __name__ == '__main__':
    print("################################################################################")
    print("This module accepts a directory over command line input and ")
    print("(1) marks all the files in the directory as valid by computing and storing the cyprotgraphic value for them")
    print("(2) verifies that all the file in the directory are authentic against the computed cryptographic value")
    print("Note: Cryptographic value is vague however it is intentionally written this way to not give away too much too easily")
    print("################################################################################")
    print('')

    while True:
        print('Please input the action that you would like to perform')
        inputNr = int(input("\\\\>"))
        if inputNr <= 2:
            break
        else:
            print("Invalid input. Please choose from the above listed actions")


    while True:
        print('Please input the directory containing the training datasets')
        inputDir = input("\\\\>")
        if os.path.exists(inputDir):
            break
        else:
            print('Entered directory does not exist. Please check the entered directory and retry')

    if inputNr == 1:
        encryptFilesAndStore(inputDir)

    elif inputNr == 2:
        decryptFilesAndVerify(inputDir)

