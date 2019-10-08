import socket
class XYStage:
    """Usage:
    from XYStage import XYStage
    xy = XYStage()
    xy.move(10,20)

1.  You need to get the two controllers setup properly, using the
Windows 10 laptop.  Theoretically, these settings survive power
cycling and do not need to be changed.

2.  Get them on the proper subnet.  I do this by patching from the
wired ethernet of the computer to the "Ethernet IN" port on the front
of the black box.  Set the address of the computer to 192.168.1.90 and
submask 255.255.255.0.

3.  Confirm that you can see the two controllers:

qrt002@qrt002:~/Lin$ nmap -p 8000 192.168.1.91

Starting Nmap 6.40 ( http://nmap.org ) at 2016-09-30 15:59 CDT
Nmap scan report for 192.168.1.91
Host is up (0.00098s latency).
PORT     STATE SERVICE
8000/tcp open  http-alt

Nmap done: 1 IP address (1 host up) scanned in 0.04 seconds
qrt002@qrt002:~/Lin$ nmap -p 8000 192.168.1.92

Starting Nmap 6.40 ( http://nmap.org ) at 2016-09-30 15:59 CDT
Nmap scan report for 192.168.1.92
Host is up (0.00080s latency).
PORT     STATE SERVICE
8000/tcp open  http-alt

Nmap done: 1 IP address (1 host up) scanned in 0.04 seconds


If you do not see "open http-alt" then you need to make it so.  One
time this happened, power cycling the black box fixed this.  I think
this was because during configuration I left a session connected.

4.  The ip address for the X axis is 192.168.1.91,
and for the Y axis, 192.168.1.92.

5.  It is good manners to disable before the program ends

    """
    def __init__(self, Xip=("192.168.1.91",8000), Yip=("192.168.1.92",8000)):
        self.Xsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Xsocket.connect(Xip)
        self.Ysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Ysocket.connect(Yip)
        self.enable()
    def enable(self):
        self.Xsocket.send("ENABLE\n")
        self.Ysocket.send("ENABLE\n")
    def move(self, x, y):
        self.Xsocket.send("MOVEABS D%d F20\n"%x)
        self.Ysocket.send("MOVEABS D%d F20\n"%y)
    def disable(self):
        self.Xsocket.send("DISABLE\n")
        self.Ysocket.send("DISABLE\n")
