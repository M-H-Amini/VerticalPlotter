import paramiko
from paramiko import SSHClient
from scp import SCPClient
import threading

def ssh(host, user, password, command):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(ssh.load_system_host_keys())
    ssh.connect(host, username=user, password=password)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
    print(ssh_stdout.read().decode('ascii').strip("\n"))

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def copyfile(host, user, password, source, dest='/home/pi/Downloads/Project 4/'):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, 22, user, password)
    scp = SCPClient(client.get_transport())
    scp.put(source, recursive=True, remote_path=dest)
    scp.close()

if __name__ == '__main__':
    command = 'scp mha@localhost:D:/VerticalPlotter/a.jpg pi@192.168.1.7:/home/pi/Downloads/Project 4/'
    host, user, password, command = '192.168.1.7', 'pi', 'raspberry', command
    copyfile(host, user, password, 'd:/VerticalPlotter/a.jpg')