# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:57:19 2019

@author: ivanovn
"""

import socket
import json
import struct
from UI.JSONParser import parse as bcip_json_parse

HEADER_SZ = 4

class BCIPSocketInterface:
    
    def __init__(self,sock,addr):
        """
        Create a socket interface
        """
        self._sess_hash = {}
        self._sock = sock
        self._addr = addr
        
        self._recv = b""
        self._send = b""
        
    
    def recieve_next_client_req(self):
        """
        Wait for the next client request and store it in the recv buffer
        """
        
        packet_sz = bytearray()
        while len(packet_sz) < HEADER_SZ:
            data = self._sock.recv(HEADER_SZ)
            packet_sz.extend(data)
        
        # convert the size to an int
        packet_sz = struct.unpack('!L',packet_sz)[0]
        #print("Received packet indicating size {}".format(packet_sz))
        
        # read in the client packet payload
        packet = bytearray()
        while len(packet) < packet_sz:
            packet += self._sock.recv(min(512,packet_sz - len(packet)))
        
        # store the packet
        self._recv = packet
        #print(self._recv)
    
    def parse_client_req(self):
        """
        Process the client request and construct the return packet
        """
        
        # convert the received packet to json
        #print(self._recv.decode('utf-8'))
        in_packet = json.loads(self._recv.decode('utf-8'))
        
        # call the JSONParser
        out_packet = bcip_json_parse(in_packet,self._sess_hash)
        
        # serialize the output packet
        self._send = out_packet.encode('utf-8')
        
    def send_response(self):
        """
        Send a response to the client request
        """
        out_packet_sz = len(self._send)
        #print("returning packet of size: ",out_packet_sz)
        out_packet_header = struct.pack('!L',out_packet_sz)
        
        self._sock.sendall(out_packet_header)
        self._sock.sendall(self._send)
        #print(self._send)
        
    def end_connection(self):
        """
        Check if the latest packet was a terminate signal
        """
        in_packet = self._recv.decode('utf-8')

        if in_packet == '"terminate"':
            return True
        else:
            return False
        
    def send_end_ack(self):
        """
        Send a return packet to indicate session is ending
        """
        self._send = json.dumps({'sts' : 200}).encode('utf-8')
        self.send_response()
        
        
        
        
# setup the connection
HOST = 'localhost'
PORT = 5000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
print("waiting for response from client at port {}".format(PORT))
conn, addr = s.accept()
print("Connected by {}".format(addr))
print("Connection Established")
print("Creating the Interface object...")
socket_ui = BCIPSocketInterface(conn,addr)
print("Done... Ready to work.")

while True:
    socket_ui.recieve_next_client_req()
    if socket_ui.end_connection():
        socket_ui.send_end_ack()
        break
    
    socket_ui.parse_client_req()
    socket_ui.send_response()

conn.close()