import socket
import sys
import threading
import traceback
import os
import os.path
import shutil
import time
import zipfile

import mlsim_misc as mlsim
import misc
import ernet
import appdirs



def run_command(conn, address):
    def exit():
        conn.send('1'.encode())  # kill signal
        exitconfirmation = conn.recv(20).decode()
        if exitconfirmation == '1':
            return True
        else:
            return False

    try:
        misc.log('connection from %s' % str(address))

        # receive data stream. it won't accept data packet greater than 2048 bytes
        data = conn.recv(2048).decode()
        misc.log('received data: %s' % data)
        argin = data.split('\n')
        if len(argin) < 2:
            return

        cmd = argin[0]
        args = argin[1:]
        if cmd == 'GetThumb':
            misc.log('GetThumb: %s' % str(data))
            res = misc.GetThumb(args)
            conn.send(res.encode())
            conn.recv(20).decode()  # ready to exit
            if exit():
                print('Can now close thread')
        elif cmd == 'Plugin_MLSIM':
            misc.log('ML-SIM Reconstruct: %s' % str(data))
            exportdir = args[0]
            
            filepaths = []
            while True:
                conn.send('p'.encode())  # send paths confirmations
                data = conn.recv(2048)
                if chr(data[0]) == 'x':  # decodes the ASCII code
                    print('received all paths', len(filepaths))
                    break
                else:
                    # equivalent to chr(10), i.e. ASCII code 10
                    res_chunk = data.split('\n'.encode())
                    # filepaths.extend(res_chunk)
                    for e in res_chunk:
                        filepaths.append(e.decode())
            
            if len(filepaths) == 0:
                if exit():
                    print('Can now close thread')
            else:
                try:
                    misc.log('now calling recon')
                    reconpaths = mlsim.reconstruct(exportdir,filepaths,conn)
                    misc.log('sending back %s' % reconpaths)
                    conn.send(('2' + '\n'.join(reconpaths)).encode())
                except:
                    errmsg = traceback.format_exc()
                    misc.log("Error in reconstruct %s" % errmsg)
                    conn.send(('2' + '\n'.join([])).encode())

                conn.recv(20).decode()  # ready to exit
                if exit():
                    print('Can now close thread')                
        elif cmd == 'Plugin_MLSIM_MM_integration':
            misc.log('MLSIM Micromanager server toggled')
            desiredState = args[0]
            port = int(args[1])
            mlsim.handle_microManagerPluginState(desiredState, port)

        elif cmd == 'Plugin_ERNet':
            misc.log('Plugin_ERNet: %s' % str(data))
            exportdir = args[0]
            weka_colours = args[1] == 'true'
            stats_tubule_sheet = args[2] == 'true'
            graph_metrics = args[3] == 'true'
            save_in_original_folders = args[4] == 'true'
            
            filepaths = []
            while True:
                conn.send('p'.encode())  # send paths confirmations
                data = conn.recv(2048)
                if chr(data[0]) == 'x':  # decodes the ASCII code
                    print('received all paths', len(filepaths))
                    break
                else:
                    # equivalent to chr(10), i.e. ASCII code 10
                    res_chunk = data.split('\n'.encode())
                    # filepaths.extend(res_chunk)
                    for e in res_chunk:
                        filepaths.append(e.decode())
            
            if len(filepaths) == 0:
                if exit():
                    print('Can now close thread')
            else:
                try:
                    outpaths = ernet.segment(exportdir,filepaths,conn,weka_colours,stats_tubule_sheet,graph_metrics,save_in_original_folders)
                    misc.log('sending back %s' % outpaths)
                    conn.send(('2' + '\n'.join(outpaths)).encode())
                except:
                    errmsg = traceback.format_exc()
                    misc.log("Error in reconstruct %s" % errmsg)
                    conn.send(('2' + '\n'.join([])).encode())

                conn.recv(20).decode()  # ready to exit
                if exit():
                    print('Can now close thread')                   
        elif cmd == 'exportToFolder' or cmd == 'exportToZip':
            print('export to folder', args[1:])
            if len(args) < 3:
                if exit():
                    print('Can now close thread')

            exportdir = args[0]
            files = args[1:]

            newdir = os.path.join(exportdir, 'Mambio Export ' +
                                  time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(newdir, exist_ok=True)

            for file in files:
                basename = os.path.basename(file)
                outfile = os.path.join(newdir, basename)
                file_exists = os.path.isfile(outfile)
                countidx = 2
                while file_exists:
                    rootname, ext = os.path.splitext(basename)
                    newbasename = '%s_%d.%s' % (rootname, countidx, ext)
                    outfile = os.path.join(newdir, newbasename)
                    file_exists = os.path.isfile(outfile)
                    countidx += 1
                shutil.copy2(file, outfile)

            if cmd == 'exportToZip':
                zipname = os.path.join(
                    exportdir, 'ML-SIM Export ' + time.strftime("%Y%m%d-%H%M%S") + '.zip')
                zippath = os.path.join(exportdir, zipname)
                zipf = zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED)
                for root, dirs, files in os.walk(newdir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
                zipf.close()
                shutil.rmtree(newdir, ignore_errors=True)

                conn.send(('z%s' % zippath).encode())  # file window can open
            else:
                conn.send(('e%s' % newdir).encode())  # file window can open

            conn.recv(20).decode()  # ready to exit
            if exit():
                print('Can now close thread')
        else:  # cmd unknown
            if exit():
                print('Can now close thread')

        conn.close()  # close the connection
        misc.log('exited thread %s' % str(address))
    except Exception as e:
        errmsg = traceback.format_exc()
        misc.log(errmsg)
        # send_misc.log(errmsg)



def socketserver():
    misc.log('starting socketserver')
    host = 'localhost'
    port = int(sys.argv[1])

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    while True:
        print('now listening')
        conn, address = server_socket.accept()  # accept new connection
        th = threading.Thread(target=run_command, args=(conn, address))
        th.daemon = True
        th.start()


import matplotlib.pyplot as plt

if __name__ == '__main__':

    misc.log(os.getcwd())
    if len(sys.argv) > 2:
        misc.SetUseCloud(sys.argv[2])

    # if no argument given # cachefolder = '/Users/cc/ML-SIM/Library/tempresize' # if no argument given
    cachefolder = '%s/Mambio-Library/tempresize' % appdirs.user_cache_dir('Mambio','Mambio')

    if len(sys.argv) > 3:
        cachefolder = sys.argv[3]

    # first time launching new version? clean up cache
    misc.SetCachefolder(cachefolder)
    os.makedirs(cachefolder, exist_ok=True)

    idx = cachefolder.split('/Mambio/')[-1]
    if idx.isdigit():
        idx = int(idx)
    else:
        idx = cachefolder.split('\\Mambio\\')[-1]
        if idx.isdigit():
            idx = int(idx)
        else:
            idx = 0  # don't clean up

    for i in range(1, idx):
        oldfolder = cachefolder.replace(
            '/Mambio/' + str(idx), '/Mambio/' + str(i))
        if os.path.isdir(oldfolder):
            shutil.rmtree(oldfolder, ignore_errors=True)

    socketserver()
